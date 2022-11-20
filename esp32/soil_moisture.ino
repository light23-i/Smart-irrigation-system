#include "WiFi.h"
#include "ESPAsyncWebServer.h"

// Set your access point network credentials
const char* ssid = "test1";
const char* password = "123456789";

// Create AsyncWebServer object on port 80
AsyncWebServer server(80);

#define SOIL_MOSITURE_SENSOR 34

int sensorValue;

String readMoisture(){
  digitalWrite(27,0);
  sensorValue = analogRead(33);
  Serial.println(sensorValue);
  return String(sensorValue);
}

void setup() {
  // put your setup code here, to run once:
   Serial.begin(9600);
  Serial.println();

// Setting the ESP as an access point
  Serial.print("Setting AP (Access Point)…");
  // Remove the password parameter, if you want the AP (Access Point) to be open
  WiFi.softAP(ssid, password);

  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);


  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    request->send_P(200, "text/plain", "Hello");
  });

  server.on("/soil", HTTP_GET, [](AsyncWebServerRequest *request){
    Serial.print("Request");
    request->send_P(200, "text/plain", readMoisture().c_str());
  });
  server.begin();
}

void loop() {

}