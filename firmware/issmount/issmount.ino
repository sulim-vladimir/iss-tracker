// ISS tracking mount firmware: Arduino Uno + 2x DRV8825.
// Pinout matches the original serialSpeed.ino wiring.
//
// Velocity-mode dual-axis stepper driver. Both axes are stepped from a single Timer1
// interrupt using phase accumulators (DDA), so any rate from ~0.005 to MAX_ISR_RATE
// steps/s is produced without per-step timer reprogramming. The host sends signed target
// rates; the firmware slews to them with a per-axis acceleration limit and reports step
// counts back. A watchdog decelerates both axes to zero if the host goes silent.
//
// Serial protocol (115200 8N1, ASCII, one command per line, '\n' terminated):
//   R <rate1> <rate2>   set target rates in steps/s (float, signed)  -> P line
//   Q                   query                                        -> P line
//   A <acc1> [<acc2>]   acceleration limits, steps/s^2               -> OK
//   M <maxrate>         max |rate|, steps/s                          -> OK
//   U <ms1> <ms2>       microstepping 1..32 (only while stopped)     -> OK
//   E <0|1>             disable/enable drivers (if ENABLE_PIN wired) -> OK
//   I <ms>              auto-disable drivers after this long idle (0 = never) -> OK
//   Z [<pos1> <pos2>]   set step counters (default 0 0)              -> OK
//   S                   decelerate to stop                           -> P line
//   X                   emergency stop (no ramp)                     -> P line
//   V                   version                                      -> ISSMOUNT <ver>
// P line: "P <pos1> <pos2> <rate1> <rate2> <millis>"
//   positions are in steps at the current microstep setting, rates in steps/s.
// Errors: "ERR <text>"

#include <Arduino.h>
#include <stdlib.h>

#define FW_VERSION "1"

// ---- pins: axis1 = RA, axis2 = Dec ----
#define AX1_STEP_PIN 3   // PD3
#define AX1_DIR_PIN  2   // PD2
#define AX2_STEP_PIN 7   // PD7
#define AX2_DIR_PIN  8   // PB0
const uint8_t MS_PINS[2][3] = {{4, 5, 6}, {9, 10, 11}};  // M0, M1, M2
#define ENABLE_PIN 12   // wire both DRV8825 ~EN pins here (active LOW). Comment out if not wired.

#define AX1_STEP_BIT _BV(PD3)
#define AX2_STEP_BIT _BV(PD7)

// ---- timing ----
#define ISR_HZ 20000UL                // Timer1 CTC rate
#define MAX_ISR_RATE 9500.0           // hard cap: need >= 2 ticks per step (high + low)
#define RAMP_PERIOD_US 1000UL         // acceleration update period
#define WATCHDOG_MS 500UL             // stop if no R/Q command within this time
#define DEFAULT_MICROSTEP 16
const double INC_PER_STEP_RATE = 4294967296.0 / ISR_HZ;

struct Axis {
  volatile uint32_t acc;       // phase accumulator
  volatile uint32_t inc;       // accumulator increment per tick (|rate| scaled)
  volatile int8_t   wantDir;   // +1 / -1 requested
  volatile int8_t   pinDir;    // direction currently on the DIR pin
  volatile bool     stepHigh;  // STEP pin is high, lower it next tick
  volatile int32_t  pos;       // step counter
  double rate;                 // current commanded rate (main loop only)
  double target;               // target rate (main loop only)
  uint8_t microstep;
};

Axis ax[2];
double accelLimit[2] = {4000.0, 4000.0};  // steps/s^2
double maxRate = 6000.0;                  // steps/s
unsigned long lastCmdMs = 0;
unsigned long lastRampUs = 0;
unsigned long idleDisableMs = 0;   // 0 = keep the drivers powered (full holding torque)
unsigned long lastMotionMs = 0;
bool driversOn = false;

void setDrivers(bool on) {
#ifdef ENABLE_PIN
  digitalWrite(ENABLE_PIN, on ? LOW : HIGH);
#endif
  driversOn = on;
}

static inline void setDirPin(uint8_t i, int8_t d) {
  if (i == 0) {
    if (d > 0) PORTD |= _BV(PD2); else PORTD &= ~_BV(PD2);
  } else {
    if (d > 0) PORTB |= _BV(PB0); else PORTB &= ~_BV(PB0);
  }
}

ISR(TIMER1_COMPA_vect) {
  // Lower any step pulses raised on the previous tick (pulse width = 50 us).
  uint8_t lower = 0;
  if (ax[0].stepHigh) { lower |= AX1_STEP_BIT; ax[0].stepHigh = false; }
  if (ax[1].stepHigh) { lower |= AX2_STEP_BIT; ax[1].stepHigh = false; }
  if (lower) PORTD &= ~lower;

  uint8_t raise = 0;
  for (uint8_t i = 0; i < 2; i++) {
    Axis &a = ax[i];
    if (a.wantDir != a.pinDir) {
      // Change DIR and delay stepping by one tick to respect DRV8825 setup time.
      setDirPin(i, a.wantDir);
      a.pinDir = a.wantDir;
      continue;
    }
    uint32_t prev = a.acc;
    a.acc = prev + a.inc;
    if (a.acc < prev) {  // overflow -> one step
      raise |= (i == 0) ? AX1_STEP_BIT : AX2_STEP_BIT;
      a.stepHigh = true;
      a.pos += a.pinDir;
    }
  }
  if (raise) PORTD |= raise;
}

void applyRate(uint8_t i) {
  Axis &a = ax[i];
  double r = a.rate;
  uint32_t inc = (uint32_t)(fabs(r) * INC_PER_STEP_RATE);
  int8_t dir = (r < 0) ? -1 : 1;
  uint8_t sreg = SREG;
  cli();
  a.inc = inc;
  if (inc != 0) a.wantDir = dir;
  SREG = sreg;
}

void rampAxes(double dt) {
  for (uint8_t i = 0; i < 2; i++) {
    Axis &a = ax[i];
    double dv = accelLimit[i] * dt;
    double err = a.target - a.rate;
    if (err > dv) a.rate += dv;
    else if (err < -dv) a.rate -= dv;
    else a.rate = a.target;
    applyRate(i);
  }
}

bool setMicrostep(uint8_t i, long ms) {
  uint8_t bits;
  switch (ms) {
    case 1: bits = 0b000; break;
    case 2: bits = 0b001; break;
    case 4: bits = 0b010; break;
    case 8: bits = 0b011; break;
    case 16: bits = 0b100; break;
    case 32: bits = 0b101; break;
    default: return false;
  }
  for (uint8_t b = 0; b < 3; b++) digitalWrite(MS_PINS[i][b], (bits >> b) & 1);
  if (ax[i].microstep) {
    // keep the counter in the same physical units
    uint8_t sreg = SREG;
    cli();
    ax[i].pos = (int32_t)((int64_t)ax[i].pos * ms / ax[i].microstep);
    SREG = sreg;
  }
  ax[i].microstep = ms;
  return true;
}

void sendP() {
  int32_t p1, p2;
  uint8_t sreg = SREG;
  cli();
  p1 = ax[0].pos;
  p2 = ax[1].pos;
  SREG = sreg;
  Serial.print(F("P "));
  Serial.print(p1);
  Serial.print(' ');
  Serial.print(p2);
  Serial.print(' ');
  Serial.print(ax[0].rate, 2);
  Serial.print(' ');
  Serial.print(ax[1].rate, 2);
  Serial.print(' ');
  Serial.println(millis());
}

double clampRate(double r) {
  if (r > maxRate) return maxRate;
  if (r < -maxRate) return -maxRate;
  return r;
}

bool parse2(char *p, double &a, double &b) {
  char *end;
  a = strtod(p, &end);
  if (end == p) return false;
  p = end;
  b = strtod(p, &end);
  return end != p;
}

void handleLine(char *line) {
  char *p = line + 1;
  char *end;
  switch (line[0]) {
    case 'R': {
      double r1, r2;
      if (!parse2(p, r1, r2)) { Serial.println(F("ERR args")); return; }
      ax[0].target = clampRate(r1);
      ax[1].target = clampRate(r2);
      lastCmdMs = millis();
      if (r1 != 0 || r2 != 0) {
        lastMotionMs = lastCmdMs;
        if (!driversOn) setDrivers(true);   // motion wanted: power up before stepping
      }
      sendP();
      break;
    }
    case 'Q':
      lastCmdMs = millis();
      sendP();
      break;
    case 'A': {
      double a1 = strtod(p, &end);
      if (end == p || a1 <= 0) { Serial.println(F("ERR args")); return; }
      p = end;
      double a2 = strtod(p, &end);
      if (end == p || a2 <= 0) a2 = a1;
      accelLimit[0] = a1;
      accelLimit[1] = a2;
      Serial.println(F("OK"));
      break;
    }
    case 'M': {
      double m = strtod(p, &end);
      if (end == p || m <= 0) { Serial.println(F("ERR args")); return; }
      maxRate = (m > MAX_ISR_RATE) ? MAX_ISR_RATE : m;
      Serial.println(F("OK"));
      break;
    }
    case 'U': {
      double m1, m2;
      if (!parse2(p, m1, m2)) { Serial.println(F("ERR args")); return; }
      if (ax[0].rate != 0 || ax[1].rate != 0 || ax[0].target != 0 || ax[1].target != 0) {
        Serial.println(F("ERR moving"));
        return;
      }
      if (!setMicrostep(0, (long)m1) || !setMicrostep(1, (long)m2)) { Serial.println(F("ERR microstep")); return; }
      Serial.println(F("OK"));
      break;
    }
    case 'E': {
      long e = strtol(p, &end, 10);
      if (end == p) { Serial.println(F("ERR args")); return; }
      setDrivers(e != 0);
      lastMotionMs = millis();
      Serial.println(F("OK"));
      break;
    }
    case 'I': {
      double ms = strtod(p, &end);
      if (end == p || ms < 0) { Serial.println(F("ERR args")); return; }
      idleDisableMs = (unsigned long)ms;
      lastMotionMs = millis();
      Serial.println(F("OK"));
      break;
    }
    case 'Z': {
      long z1 = strtol(p, &end, 10);
      long z2 = 0;
      if (end != p) { p = end; z2 = strtol(p, &end, 10); } else { z1 = 0; }
      uint8_t sreg = SREG;
      cli();
      ax[0].pos = z1;
      ax[1].pos = z2;
      SREG = sreg;
      Serial.println(F("OK"));
      break;
    }
    case 'S':
      ax[0].target = 0;
      ax[1].target = 0;
      sendP();
      break;
    case 'X':
      for (uint8_t i = 0; i < 2; i++) { ax[i].target = 0; ax[i].rate = 0; applyRate(i); }
      sendP();
      break;
    case 'V':
      Serial.println(F("ISSMOUNT " FW_VERSION));
      break;
    default:
      Serial.println(F("ERR cmd"));
  }
}

void setup() {
  pinMode(AX1_STEP_PIN, OUTPUT);
  pinMode(AX2_STEP_PIN, OUTPUT);
  pinMode(AX1_DIR_PIN, OUTPUT);
  pinMode(AX2_DIR_PIN, OUTPUT);
  for (uint8_t i = 0; i < 2; i++)
    for (uint8_t b = 0; b < 3; b++) pinMode(MS_PINS[i][b], OUTPUT);
#ifdef ENABLE_PIN
  pinMode(ENABLE_PIN, OUTPUT);
#endif
  setDrivers(false);  // stay quiet and cool until the host asks for motion

  for (uint8_t i = 0; i < 2; i++) {
    ax[i].acc = 0; ax[i].inc = 0; ax[i].wantDir = 1; ax[i].pinDir = 1;
    ax[i].stepHigh = false; ax[i].pos = 0; ax[i].rate = 0; ax[i].target = 0;
    ax[i].microstep = 0;
    setDirPin(i, 1);
    setMicrostep(i, DEFAULT_MICROSTEP);
  }

  cli();
  TCCR1A = 0;
  TCCR1B = _BV(WGM12) | _BV(CS11);        // CTC, prescaler 8 -> 2 MHz
  OCR1A = (F_CPU / 8 / ISR_HZ) - 1;        // 99 -> 20 kHz
  TIMSK1 = _BV(OCIE1A);
  sei();

  Serial.begin(115200);
  Serial.println(F("ISSMOUNT " FW_VERSION));
  lastRampUs = micros();
  lastCmdMs = millis();
}

char buf[48];
uint8_t bufLen = 0;

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (bufLen) { buf[bufLen] = 0; handleLine(buf); bufLen = 0; }
    } else if (bufLen < sizeof(buf) - 1) {
      buf[bufLen++] = c;
    } else {
      bufLen = 0;
      Serial.println(F("ERR overflow"));
    }
  }

  if (millis() - lastCmdMs > WATCHDOG_MS) {
    ax[0].target = 0;
    ax[1].target = 0;
  }

  // Drop the holding current once the axes have been still for a while: the coils sing at
  // microstep holding current, and an idle mount does not need the torque.
  if (ax[0].rate != 0 || ax[1].rate != 0) lastMotionMs = millis();
  if (driversOn && idleDisableMs && millis() - lastMotionMs > idleDisableMs) setDrivers(false);

  unsigned long now = micros();
  unsigned long el = now - lastRampUs;
  if (el >= RAMP_PERIOD_US) {
    lastRampUs = now;
    rampAxes(el * 1e-6);
  }
}
