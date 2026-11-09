// ESP32 DevKit (original ESP32): GPIO34 ADC1 input.
// ONLY connect a safely scaled, isolated/test signal within ADC limits.
const int EMG_PIN = 34;
const unsigned long SAMPLE_US = 10000; // 100 Hz for smoothed envelope
unsigned long nextSample;
void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(EMG_PIN, ADC_11db);
  nextSample = micros();
}
void loop() {
  if ((int32_t)(micros() - nextSample) >= 0) {
    nextSample += SAMPLE_US;
    int adc = analogRead(EMG_PIN);
    int mv = analogReadMilliVolts(EMG_PIN); // approximate calibrated ADC-pin voltage
    Serial.printf("%lu,%d,%d\n", millis(), adc, mv);
  }
}
