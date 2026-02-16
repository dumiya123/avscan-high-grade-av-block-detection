# AV Block Detection Prototype Demonstration Script 🎥
## IPD Submission - 10 Minute Demo Guide

**Time Allocation:**
- Introduction: 1:00
- Home & Navigation: 1:00
- File Upload & Preview: 2:00
- AI Analysis & XAI: 4:00
- Report Generation & Conclusion: 2:00

---

### 1. START: Introduction (0:00 - 1:00)

**[Screen: Showing the desktop with your project code open in VS Code, then switch to the browser Landing Page]**

**Speaker:**
"Hello, this is the demonstration for my IPD project, **AtrionNet**, an Automated AV Block Detection System using Deep Learning.

The problem I am addressing is the complexity and time required to manually diagnose Atrioventricular (AV) blocks from ECG signals. My solution uses a multi-task U-Net architecture to automatically segment ECG waves and classify AV blocks with high accuracy.

Today, I will show you the working prototype, demonstrating the full end-to-end flow from uploading a raw ECG signal to generating a clinical diagnostic report. Please note that this is a working prototype, so while the core logic is fully functional, we are still refining some UI elements."

---

### 2. Home Page & Branding (1:00 - 2:00)

**[Screen: AtrionNet Landing Page]**

**Action:**
*   **Mouse over** the "AtrionNet" logo.
*   **Scroll down** slowly to show the features section.
*   **Point to** the "Get Started" button.

**Speaker:**
"Here is the landing page. I designed this with a focus on medical professionalism and clean aesthetics.
As you can see, the interface is simple and distraction-free, which is critical for clinical environments. We have clear distinct sections explaining the system's capabilities—Rapid Analysis, 99% Accuracy, and Detailed Reporting."

**Action:**
*   **Click** on the "Instructions" link in the navigation bar.

**Speaker:**
"Before starting an analysis, the user can check the Instructions page. This guides medical personnel on the correct file formats and how to interpret the results, ensuring the tool is used correctly."

**Action:**
*   **Click** back to "Home" or "Get Started".

---

### 3. File Upload & Clinical Preview (2:00 - 4:00)

**[Screen: Analysis Page - Initial State]**

**Action:**
*   **Click** "Get Started" to enter the main Analysis Dashboard.
*   **Point to** the "Upload ECG File" area.

**Speaker:**
"This is the main workspace. It is designed with a two-stage workflow to mimic how doctors actually work: first a quick look, then a deep analysis.

I'll start by uploading a raw ECG file. The system accepts standard `.npy` numpy files which are common in medical research datasets."

**Action:**
*   **Click** "Select File" and choose a sample file (e.g., a file showing a specific block layout).
*   **Wait** for the loading spinner.
*   **Point to** the "Clinical Preview" grid that appears on the LEFT side.

**Speaker:**
"Once the file is uploaded, the system immediately processes it for the **Clinical Preview**.
This is a crucial feature. Notice the red grid background—this exactly replicates the physical ECG paper doctors use.
*   Small squares represent 0.04 seconds.
*   Large squares represent 0.2 seconds.

This allows a cardiologist to do a 'sanity check' on the signal quality before asking the AI for a diagnosis. It builds trust in the system."

---

### 4. AI Analysis & Explainability (4:00 - 8:00)

**[Screen: Analysis Page - Split View]**

**Action:**
*   **Point to** the "Analyze ECG" button in the center.
*   **Click** the button.
*   **Show** the loading state (mentioning the backend is running the U-Net model).

**Speaker:**
"Now, I will run the Deep Learning engine. When I click 'Analyze', the data is sent to my FastAPI backend.
Inside the backend, my **Multi-Task U-Net model** is performing two jobs simultaneously:
1.  **Segmentation:** It is finding the exact start and end of P-waves, QRS complexes, and T-waves.
2.  **Classification:** It is looking for the specific patterns of AV blocks."

**Action:**
*   **Wait** for the "AI Diagnostic Result" to appear on the RIGHT side.
*   **Point to** the **Diagnosis Title** (e.g., "3rd Degree AV Block").
*   **Point to** the **Confidence Score** badge.

**Speaker:**
"Here is the result. The system has detected [Say the diagnosis, e.g., a 3rd Degree AV Block] with [Say confidence, e.g., 96%] confidence.

But a simple label isn't enough for medical safety. We need to know *why*."

**Action:**
*   **Mouse over** the XAI (Explainable AI) visualization on the right.
*   **Point to** the colored heatmap overlays (Red/Blue areas).

**Speaker:**
"This is the **Explainable AI (XAI)** module, specifically using **Grad-CAM**.
The colored heatmap you see overlaid on the signal shows exactly where the model 'looked' to make this decision.
*   The **Red areas** indicate high attention—notice how it is focusing on the P-waves and the gaps between waves.
*   This proves the model isn't just guessing; it's looking at the clinically relevant features, just like a human doctor would."

**Action:**
*   **Scroll down** slightly to show the calculated metrics (Intervals).

**Speaker:**
"Below the visual, we also extract the raw clinical metrics: Heart Rate, PR Intervals, and QRS duration. These calculated values serve as a second layer of validation for the doctor."

---

### 5. Reporting & Conclusion (8:00 - 10:00)

**[Screen: Analysis Page - Bottom Action Bar]**

**Action:**
*   **Point to** the "Download Report" button.
*   **Click** it.
*   **Open** the downloaded PDF file on screen.

**Speaker:**
"Finally, for clinical documentation, we can generate a medical report.
This PDF includes:
1.  Patient ID and Timestamp.
2.  The full ECG strip with the diagnosis.
3.  The XAI insights and confidence scores.

This document can be directly attached to a patient's electronic health record."

**[Screen: Back to the Landing Page or a slide showing "Future Work"]**

**Speaker:**
"So, looking at the progress:
I have successfully implemented the full pipeline: Data Ingestion → Preprocessing → Deep Learning Analysis → XAI Visualization → Reporting.

The core 'Logic Tier' and 'Presentation Tier' are functioning well.
My remaining work focuses on training the model on a larger dataset to improve robustness for rare cases, and adding support for multi-lead ECGs.

This prototype demonstrates a functional, end-to-end solution for automated AV block detection that prioritizes both accuracy and clinical interpretability.

Thank you for your time."

---
