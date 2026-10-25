# OptiScan-Agent: 5-Minute Video Presentation Script
### OpenCV AI Competition 2026 Powered by AWS
**Target Length:** 4 minutes 30 seconds (Max allowed: 5 minutes)  
**Recording Tools Suggested:** OBS Studio, Loom, or Windows Game Bar (Win+G)  

---

### [0:00 - 0:40] Problem & Introduction (Cam + Slide / Title Screen)
* **Visual:** Your webcam showing you / your team, or title screen showing OptiScan-Agent banner.
* **Spoken Script:**
  > "Hello OpenCV and AWS judges! My name is [Your Name], and I am proud to present **OptiScan-Agent**: an autonomous closed-loop industrial inspection system powered by OpenCV 5 and AWS.
  >
  > In high-speed electronics and semiconductor manufacturing, standard Automated Optical Inspection systems suffer from a major bottleneck: they are **static, single-pass setups**. When factory lighting glares off a solder joint, or when a hairline crack is too small to resolve under fixed focus, static systems either produce false alarms—costing millions in manual review—or miss defects completely.
  >
  > We built OptiScan-Agent to fulfill the core mission of Physical and Agentic AI: a system where visual perception doesn't just describe an image, but **directly drives subsequent plans, tool calls, dynamic re-perception, and physical machine actuation**."

---

### [0:40 - 1:30] Architecture & Technical Design (Show Architecture Diagram / Slide)
* **Visual:** Show the Architecture Flow diagram from `TECHNICAL_REPORT.md` or Tab 4 of the Streamlit dashboard.
* **Spoken Script:**
  > "OptiScan-Agent is built on a tight closed loop:
  > First, our OpenCV 5 vision pipeline captures a wide-field survey frame, performing geometric alignment using ORB homography and large-kernel Gaussian background subtraction to cancel glare.
  >
  > Next, the engine extracts candidate anomaly regions and computes quantitative uncertainty metrics: Shannon visual entropy and Sobel gradient energy.
  >
  > This is where our agent orchestrator—integrated with AWS Bedrock—takes over. When ambiguity is detected, the agent autonomously executes a tool call: `tool_active_micro_rescan`. It dynamically commands sub-pixel Lanczos-4 optical zoom and Contrast Limited Adaptive Histogram Equalization (CLAHE) on the targeted region.
  >
  > Finally, once the defect signature is empirically verified, the agent commands physical sorting actuators and streams telemetry to AWS DynamoDB and S3."

---

### [1:30 - 3:00] Live Demonstration (Screen Share: Streamlit Dashboard)
* **Visual:** Switch to browser showing `http://localhost:8501`.
* **Action 1 (Glare Suppression):**
  - Select *"Nominal Board with Industrial Lighting Glare"*. Click *"Trigger Autonomous Inspection Loop"*.
  - Show how the background high-pass filter and active re-scan eliminate the false alarm.
* **Spoken Script:**
  > "Let's see OptiScan-Agent in action. Here we have a nominal circuit board subjected to harsh overhead industrial glare. In a traditional static AOI system, this glare hotspot triggers a false reject. 
  > But watch what OptiScan-Agent does: the agent recognizes the low structural residual under illumination normalization, suppresses the false alarm, approves the board with 99% confidence, and directs the conveyor to PASS."

* **Action 2 (IC Solder Bridge & Active Re-Scan):**
  - Select *"IC Solder Bridge (Electrical Short Risk)"*. Click *"Trigger Autonomous Inspection Loop"*.
  - Point out Column 1 (Survey Scan yellow box) $\to$ Column 2 (High-Res Zoomed Crop with CLAHE) $\to$ Column 3 (Physical Actuation: `DIVERT_REWORK_BIN_A`).
* **Spoken Script:**
  > "Now, let's load a critical defect: an IC solder bridge between micro-pins under glare.
  > In the survey scan, the anomaly is flagged with borderline confidence because glare obscures the pin boundary.
  > Notice what happens next: the agent's reasoning trace logs that visual evidence is ambiguous, so it invokes `tool_active_micro_rescan` with a 3.5x optical zoom and HDR-CLAHE.
  > OpenCV 5 re-scans the ROI, reveals the sub-pixel bridge spanning the pins, and computes an edge density of 0.11. The agent confirms the electrical short with 97% confidence and immediately fires the pneumatic diverter to route the board to Rework Bin A."

---

### [3:00 - 3:45] Agentic Trace & Observability (Click "Observable Decision Trace" Tab)
* **Visual:** Open Tab 2 (Observable Agentic Decision Trace). Scroll through the step-by-step audit log.
* **Spoken Script:**
  > "In Tab 2, we can inspect the full observable audit log. Every step—from preliminary perception, to agent reasoning, tool call parameterization, active re-perception, and final relay actuation—is recorded with microsecond timestamps and telemetry payloads.
  > If an unidentifiable foreign object is encountered, the agent safely escalates the case to a senior QA engineer with annotated evidence, ensuring robust human-in-the-loop governance."

---

### [3:45 - 4:15] AWS Graviton4 + COOL Benchmarks (Click Tab 4)
* **Visual:** Open Tab 4 (AWS Graviton & COOL Architecture). Highlight the comparative metrics table.
* **Spoken Script:**
  > "For cloud delivery and the Best Use of COOL Award, we evaluated OptiScan-Agent on AWS Graviton4 c8g.xlarge instances running the Cloud-Optimized OpenCV Library with ARM NEON and SVE2 acceleration.
  > Compared to baseline x86 instances, Graviton4 + COOL achieves a 2.19x reduction in latency—processing cycles in under 28 milliseconds—boosting throughput by 119% and cutting AWS compute costs by 60.1% per million inspections."

---

### [4:15 - 4:30] Conclusion & Call to Action (Cam / Title Screen)
* **Visual:** Return to camera or title screen.
* **Spoken Script:**
  > "OptiScan-Agent proves that combining OpenCV 5's mathematical vision algorithms with AWS Bedrock agentic orchestration and Graviton compute transforms vision from passive observation into active, reliable decision-making.
  >
  > All code, Docker configurations, and benchmarks are fully reproducible in our public repository. Thank you, OpenCV and AWS, for this incredible competition!"
