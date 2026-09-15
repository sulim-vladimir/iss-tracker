#include <AccelStepper.h>

#define STP0 3
#define DIR0 2
#define M00 4
#define M01 5
#define M02 6
#define CONTROL0 A0

#define STP1 7
#define DIR1 8
#define M10 9
#define M11 10
#define M12 11
#define CONTROL1 A1

#define SPEED_LIMIT 4.0
#define DEFAULT_SPEED 0.5
#define HIGH_SPEED_THRESHOLD 1.5

#define HIGH_SPEED_MICROSTEP_RA 2
#define LOW_SPEED_MICROSTEP_RA 4
#define HIGH_SPEED_MICROSTEP_DE 4
#define LOW_SPEED_MICROSTEP_DE 8

#define STEP_DEG 1.8
#define PULLEY_RATIO 2.25
#define WORM_GEAR_RATIO_RA 130
#define WORM_GEAR_RATIO_DE 65

AccelStepper stepperRa(AccelStepper::DRIVER, STP0, DIR0);
AccelStepper stepperDe(AccelStepper::DRIVER, STP1, DIR1);

unsigned long previousMillis = 0;
bool slewing = false;
int microstepRa;
int microstepDe;

void setMicrostep(int m0, int m1, int m2, int microstep)
{
  switch (microstep) {
    case 2:
      digitalWrite(m0, HIGH);   
      digitalWrite(m1, LOW);
      digitalWrite(m2, LOW);     
      break;
    case 4:
      digitalWrite(m0, LOW);   
      digitalWrite(m1, HIGH);
      digitalWrite(m2, LOW);     
      break;
    case 8:
      digitalWrite(m0, HIGH);   
      digitalWrite(m1, HIGH);
      digitalWrite(m2, LOW);     
      break;
    default:
      return;
  }
}

void setMicrostepRa(int microstep)
{
  Serial.print("speedStepSecRa: ");
  Serial.println(microstep);
  setMicrostep(M00, M01, M02, microstep);
  microstepRa = microstep;
}

void setMicrostepDe(int microstep)
{
  Serial.print("speedStepSecDe: ");
  Serial.println(microstep);
  setMicrostep(M10, M11, M12, microstep);  
  microstepDe = microstep;
}

int speedStepSecRa(float speed, int microstep)
{
  return speed / STEP_DEG * WORM_GEAR_RATIO_RA * PULLEY_RATIO * microstep;
}

int speedStepSecDe(float speed, int microstep)
{
  return speed / STEP_DEG * WORM_GEAR_RATIO_DE * PULLEY_RATIO * microstep;
}

void setSpeedRa(float speed) 
{
  if (speed > SPEED_LIMIT) {
    speed = SPEED_LIMIT;
  }

  if (speed > HIGH_SPEED_THRESHOLD) {
    if (microstepRa != HIGH_SPEED_MICROSTEP_RA) {
      setMicrostepRa(HIGH_SPEED_MICROSTEP_RA);
    }
  } else {
    if (microstepRa != LOW_SPEED_MICROSTEP_RA) {
      setMicrostepRa(LOW_SPEED_MICROSTEP_RA);
    }
  }

  stepperRa.setSpeed(speedStepSecRa(speed, microstepRa));
}

void setSpeedDe(float speed) 
{
  if (speed > SPEED_LIMIT) {
    speed = SPEED_LIMIT;
  }

  if (speed > HIGH_SPEED_THRESHOLD) {
    if (microstepDe != HIGH_SPEED_MICROSTEP_DE) {
      setMicrostepDe(HIGH_SPEED_MICROSTEP_DE);
    }
  } else {
    if (microstepDe != LOW_SPEED_MICROSTEP_DE) {
      setMicrostepDe(LOW_SPEED_MICROSTEP_DE);
    }
  }

  stepperDe.setSpeed(speedStepSecDe(speed, microstepDe));
}

void setup()
{  
  Serial.begin(115200);
  Serial.setTimeout(10);
  
  pinMode(M00, OUTPUT);
  pinMode(M01, OUTPUT);
  pinMode(M02, OUTPUT);
  pinMode(M10, OUTPUT);
  pinMode(M11, OUTPUT);
  pinMode(M12, OUTPUT);

  setSpeedRa(DEFAULT_SPEED);
  setSpeedDe(DEFAULT_SPEED);

  // Speed in step per sec is higher for low speed microstep
  stepperRa.setMaxSpeed(speedStepSecRa(SPEED_LIMIT, LOW_SPEED_MICROSTEP_RA));
  stepperDe.setMaxSpeed(speedStepSecDe(SPEED_LIMIT, LOW_SPEED_MICROSTEP_DE));

  stepperRa.setAcceleration(450);
  stepperDe.setAcceleration(450);
}

void loop()
{ 
  String inString;
  if (Serial.available() > 0) {
    int command = Serial.read();
    if (!slewing && command == 'J') {
      slewing = true;
      stepperRa.moveTo(Serial.parseInt());
      stepperDe.moveTo(Serial.parseInt());     
    }
    if (!slewing && command == 'j') {
      slewing = true;
      stepperRa.move(Serial.parseInt());
      stepperDe.move(Serial.parseInt());     
    }
    if (!slewing && command == 'S') {
      setSpeedRa(Serial.parseFloat());
      setSpeedDe(Serial.parseFloat());           
    }
    if (command == 's') {
      setSpeedRa(0);
      setSpeedDe(0);           
      stepperRa.stop();
      stepperDe.stop();
      slewing = false;   
    }
  }
  
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= 1000) {
    previousMillis = currentMillis;
    Serial.print(stepperRa.currentPosition());
    Serial.print(":");
    Serial.print(stepperDe.currentPosition());
    if (slewing) {
      Serial.print(" S");
    }
    Serial.println();
  }

  if (slewing) {
    if (stepperRa.distanceToGo() == 0 && stepperDe.distanceToGo() == 0) {
      slewing = false;
    }
    stepperRa.run();
    stepperDe.run();
  } else {
    stepperRa.runSpeed();
    stepperDe.runSpeed();
  }
}
