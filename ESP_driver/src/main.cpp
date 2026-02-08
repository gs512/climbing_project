// climbing_project/ESP_driver/src/main.cpp
#include <Arduino.h>
#include <vector>
#include <cstring>

#define LASER_CONTROL_PIN 2
static const uint32_t BAUD_RATE = 115200;
static const uint16_t DAC_MIN = 0;
static const uint16_t DAC_MAX = 4095;
static const unsigned long HEARTBEAT_INTERVAL_MS = 1000;

// Drawing step timing (controls speed)
static const unsigned long STEP_INTERVAL_MS = 8; // 8 ms between waypoints -> smooth & reasonably fast

// Limits
static const size_t MAX_CMD_QUEUE = 64;
static const size_t MAX_WAYPOINTS = 128;

struct Cmd {
    uint16_t x;
    uint16_t y;
    char shape; // 'C' circle, 'X' cross, 'S' square
    uint16_t size; // diameter in DAC units
};

static char lineBuffer[96];
static size_t linePos = 0;

static unsigned long lastHeartbeat = 0;
static bool laserState = false;

// Command queue
static Cmd cmdQueue[MAX_CMD_QUEUE];
static size_t queueHead = 0;
static size_t queueTail = 0;
static bool queueEmpty = true;

// Active drawing waypoints
struct WP {
    uint16_t x;
    uint16_t y;
};
static WP waypoints[MAX_WAYPOINTS];
static size_t wpCount = 0;
static size_t wpIndex = 0;
static unsigned long lastStepTime = 0;
static bool drawing = false;

// --- STUB: Replace this with your actual DAC writes ---
void writeToDAC(uint16_t x, uint16_t y) {
    // clamp
    if (x > DAC_MAX) x = DAC_MAX;
    if (y > DAC_MAX) y = DAC_MAX;
    Serial.printf("DAC_MOVE %u,%u\r\n", x, y);
}
// ----------------------------------------------------------------

// Simple circular buffer queue helpers
bool enqueueCmd(const Cmd &c) {
    size_t nextTail = (queueTail + 1) % MAX_CMD_QUEUE;
    if (nextTail == queueHead && !queueEmpty) {
        // full
        return false;
    }
    cmdQueue[queueTail] = c;
    queueTail = nextTail;
    queueEmpty = false;
    return true;
}
bool hasCmd() {
    return !queueEmpty;
}
Cmd dequeueCmd() {
    Cmd c = cmdQueue[queueHead];
    queueHead = (queueHead + 1) % MAX_CMD_QUEUE;
    if (queueHead == queueTail) queueEmpty = true;
    return c;
}

// Create waypoints for a circle (approx polygon), square, or cross
size_t build_waypoints_circle(uint16_t cx, uint16_t cy, uint16_t diameter) {
    size_t n = 20; // points around circle; tradeoff speed vs smoothness
    if (n > MAX_WAYPOINTS) n = MAX_WAYPOINTS;
    float r = diameter * 0.5f;
    for (size_t i = 0; i < n; ++i) {
        float t = (float)i / (float)n * 2.0f * 3.14159265358979323846f;
        float fx = cx + r * cos(t);
        float fy = cy + r * sin(t);
        if (fx < 0) fx = 0;
        if (fy < 0) fy = 0;
        if (fx > DAC_MAX) fx = DAC_MAX;
        if (fy > DAC_MAX) fy = DAC_MAX;
        waypoints[i].x = (uint16_t)round(fx);
        waypoints[i].y = (uint16_t)round(fy);
    }
    return n;
}

size_t build_waypoints_square(uint16_t cx, uint16_t cy, uint16_t diameter) {
    // 4 corners: draw loop
    float half = diameter * 0.5f;
    WP pts[4];
    pts[0] = {(uint16_t)max(0.0f, cx - half), (uint16_t)max(0.0f, cy - half)};
    pts[1] = {(uint16_t)min((float)DAC_MAX, cx + half), (uint16_t)max(0.0f, cy - half)};
    pts[2] = {(uint16_t)min((float)DAC_MAX, cx + half), (uint16_t)min((float)DAC_MAX, cy + half)};
    pts[3] = {(uint16_t)max(0.0f, cx - half), (uint16_t)min((float)DAC_MAX, cy + half)};
    // Expand to a small polygon sequence for smooth drawing
    size_t idx = 0;
    for (int k = 0; k < 4; ++k) {
        size_t next = (k + 1) % 4;
        // linear interpolate few points along edge
        for (int s = 0; s < 6 && idx < MAX_WAYPOINTS; ++s) {
            float alpha = (float)s / 6.0f;
            float fx = pts[k].x * (1.0f - alpha) + pts[next].x * alpha;
            float fy = pts[k].y * (1.0f - alpha) + pts[next].y * alpha;
            waypoints[idx].x = (uint16_t)round(fx);
            waypoints[idx].y = (uint16_t)round(fy);
            idx++;
        }
    }
    return idx;
}

size_t build_waypoints_cross(uint16_t cx, uint16_t cy, uint16_t diameter) {
    // Two lines crossing: we create short sequences along each line
    float half = diameter * 0.5f;
    size_t idx = 0;
    // line 1: top-left -> bottom-right
    for (int s = 0; s < 12 && idx < MAX_WAYPOINTS; ++s) {
        float alpha = (float)s / 11.0f;
        float fx = (cx - half) * (1.0f - alpha) + (cx + half) * alpha;
        float fy = (cy - half) * (1.0f - alpha) + (cy + half) * alpha;
        waypoints[idx].x = (uint16_t)round(fx);
        waypoints[idx].y = (uint16_t)round(fy);
        idx++;
    }
    // small gap then line 2: bottom-left -> top-right
    for (int s = 0; s < 12 && idx < MAX_WAYPOINTS; ++s) {
        float alpha = (float)s / 11.0f;
        float fx = (cx - half) * (1.0f - alpha) + (cx + half) * alpha;
        float fy = (cy + half) * (1.0f - alpha) + (cy - half) * alpha;
        waypoints[idx].x = (uint16_t)round(fx);
        waypoints[idx].y = (uint16_t)round(fy);
        idx++;
    }
    return idx;
}

void startDrawingFromCmd(const Cmd &c) {
    // Build waypoints depending on shape
    wpCount = 0;
    wpIndex = 0;
    if (c.shape == 'C') {
        wpCount = build_waypoints_circle(c.x, c.y, c.size);
    } else if (c.shape == 'S') {
        wpCount = build_waypoints_square(c.x, c.y, c.size);
    } else if (c.shape == 'X') {
        wpCount = build_waypoints_cross(c.x, c.y, c.size);
    } else {
        // fallback: single point
        waypoints[0].x = c.x;
        waypoints[0].y = c.y;
        wpCount = 1;
    }
    drawing = (wpCount > 0);
    lastStepTime = millis();
}

void processDrawing() {
    if (!drawing) return;
    unsigned long now = millis();
    if ((now - lastStepTime) < STEP_INTERVAL_MS) return;
    lastStepTime = now;

    if (wpIndex < wpCount) {
        writeToDAC(waypoints[wpIndex].x, waypoints[wpIndex].y);
        wpIndex++;
    } else {
        // finished current drawing
        drawing = false;
    }
}

bool parseLineToCmd(const char *line, Cmd &out) {
    // Expect format: "X,Y,TYPE,SZ"
    // Use strtok/strtol to be robust
    char tmp[96];
    strncpy(tmp, line, sizeof(tmp)-1);
    tmp[sizeof(tmp)-1] = '\0';

    char *p = strtok(tmp, ",");
    if (!p) return false;
    long lx = strtol(p, nullptr, 10);

    p = strtok(nullptr, ",");
    if (!p) return false;
    long ly = strtol(p, nullptr, 10);

    p = strtok(nullptr, ",");
    if (!p) return false;
    char typeChar = p[0];

    p = strtok(nullptr, ",");
    if (!p) return false;
    long lsz = strtol(p, nullptr, 10);

    if (lx < 0) lx = 0;
    if (ly < 0) ly = 0;
    if (lsz < 0) lsz = 1;

    if (lx > DAC_MAX) lx = DAC_MAX;
    if (ly > DAC_MAX) ly = DAC_MAX;
    if (lsz > DAC_MAX) lsz = DAC_MAX;

    out.x = (uint16_t)lx;
    out.y = (uint16_t)ly;
    out.shape = typeChar;
    out.size = (uint16_t)lsz;
    return true;
}

void handleSerialInput() {
    while (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            lineBuffer[linePos] = '\0';
            if (linePos > 0) {
                Cmd cmd;
                if (parseLineToCmd(lineBuffer, cmd)) {
                    if (!enqueueCmd(cmd)) {
                        Serial.println("WARN: cmd queue full, dropped cmd");
                    } else {
                        Serial.printf("QUEUED %u,%u,%c,%u\r\n", cmd.x, cmd.y, cmd.shape, cmd.size);
                    }
                } else {
                    Serial.printf("WARN: invalid cmd '%s'\r\n", lineBuffer);
                }
            }
            linePos = 0;
        } else {
            if (linePos < sizeof(lineBuffer)-1) {
                lineBuffer[linePos++] = c;
            } else {
                // overflow: reset
                linePos = 0;
            }
        }
    }
}

void heartbeat() {
    unsigned long now = millis();
    if (now - lastHeartbeat >= HEARTBEAT_INTERVAL_MS) {
        lastHeartbeat = now;
        laserState = !laserState;
        digitalWrite(LASER_CONTROL_PIN, laserState ? HIGH : LOW);
    }
}

void setup() {
    pinMode(LASER_CONTROL_PIN, OUTPUT);
    digitalWrite(LASER_CONTROL_PIN, LOW);
    Serial.begin(BAUD_RATE);
    unsigned long start = millis();
    while (!Serial && (millis() - start) < 3000) {
        delay(10);
    }
    Serial.println("ESP32 Laser Driver v1 - shape protocol");
}

void loop() {
    // 1) read serial into queue
    handleSerialInput();

    // 2) if not currently drawing and have queued commands, start next
    if (!drawing && hasCmd()) {
        Cmd next = dequeueCmd();
        startDrawingFromCmd(next);
    }

    // 3) step drawing
    processDrawing();

    // 4) heartbeat
    heartbeat();
}
