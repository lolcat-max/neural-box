// arduino_analog_id_streamer.ino
// Analog -> integer token ID streamer.
//
// PC -> Arduino (ASCII, newline-terminated):
//   PING
//   V<integer>   e.g. V5000
//   N<integer>   e.g. N300000
//   GO
//
// Arduino -> PC:
//   READY (repeated until configured)
//   PONG
//   OK V=...
//   OK N=...
//   START
//   <id>   (N lines)
//   DONE

const int ANALOG_PIN = A0;

// Typical AVR ADC range is 0..1023 (10-bit). [web:128]
const long ADC_MIN = 0;
const long ADC_MAX = 1023;

long V = -1; // vocab size
long N = -1; // how many ids to output

String readLine() {
  if (!Serial.available()) return String();
  String s = Serial.readStringUntil('\n');
  s.trim();
  return s;
}

long analogToToken(long adcValue, long vocabSize) {
  // map() remaps but does not constrain; clamp manually. [web:136]
  long id = map(adcValue, ADC_MIN, ADC_MAX, 0, vocabSize - 1);
  if (id < 0) id = 0;
  if (id >= vocabSize) id = vocabSize - 1;
  return id;
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(2000);

  // Don't hang forever on native USB boards.
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 2000) { }

  pinMode(ANALOG_PIN, INPUT);
}

void loop() {
  // Repeat READY until configured.
  static unsigned long lastReady = 0;
  if ((V <= 0 || N <= 0) && millis() - lastReady > 500) {
    Serial.println("READY");
    lastReady = millis();
  }

  String line = readLine();
  if (line.length() == 0) return;

  if (line == "PING") {
    Serial.println("PONG");
    return;
  }

  if (line.charAt(0) == 'V') {
    V = line.substring(1).toInt();
    Serial.print("OK V=");
    Serial.println(V);
    return;
  }

  if (line.charAt(0) == 'N') {
    N = line.substring(1).toInt();
    Serial.print("OK N=");
    Serial.println(N);
    return;
  }

  if (line == "GO") {
    if (V <= 0 || N <= 0) {
      Serial.println("ERR not configured");
      return;
    }

    Serial.println("START");

    for (long i = 0; i < N; i++) {
      long adc = analogRead(ANALOG_PIN);
      long id  = analogToToken(adc, V);
      Serial.println(id);

      // If your PC can’t keep up, uncomment:
      // delayMicroseconds(100);
    }

    Serial.println("DONE");
    while (true) { delay(1000); }
  }

  Serial.println("ERR unknown cmd");
}
