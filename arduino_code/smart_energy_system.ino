int relayPin = 7;  // Pin connected to the relay module

void setup() {
  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, LOW); // Ensure relay is off initially
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char signal = Serial.read();  // Read signal from Python script
    if (signal == '0') {
      digitalWrite(relayPin, HIGH); // Turn ON appliances
    } else if (signal == '1') {
      digitalWrite(relayPin, LOW);  // Turn OFF appliances
    }
  }
}
