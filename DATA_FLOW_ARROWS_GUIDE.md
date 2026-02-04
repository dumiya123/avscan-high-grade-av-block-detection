# Logic Tier Data Flow Arrows Guide

Complete guide for adding arrows to your draw.io diagram.

## Two Types of Arrows

### Type 1: HIGH-LEVEL (Between Modules)
- Color: Black
- Width: 3pt
- Style: Solid

### Type 2: INTERNAL (Within Modules)
- Color: Dark Gray
- Width: 2pt
- Style: Solid

## HIGH-LEVEL ARROWS (17 Total)

### 1. Proxy Server → File Upload Handler
Label: "HTTP POST Request"

### 2. Temp Storage Manager → ECG Signal Loader
Label: "File Path"

### 3. Signal Sampler → AV Block Predictor
Label: "Clean ECG Data"

### 4. U-NET Segmentation → GRAD-CAM Generator
Label: "Feature Maps"

### 5. Attention Module → Saliency MAP generator
Label: "Attention Weights"

### 6. Dissociation Detector → Clinical Rule Engine
Label: "Dissociation Ratio"

### 7. Severity Assessor → Clinical Report Builder
Label: "Diagnosis Summary"

### 8. Clinical Explain Generator → Clinical Report Builder
Label: "Explanation Text"

### 9. Wave Boundary Extractor → Wave Interval Scaler
Label: "Wave Positions"

### 10. Severity Assessor → Response Builder
Label: "Final Diagnosis"

### 11. Clinical Explain Generator → Response Builder
Label: "Explanation"

### 12. Clinical Report Builder → Response Builder
Label: "Report ID"

### 13. Response Builder → API Router
Label: "JSON Response"

### 14-17. State Management (Dashed, Gray)
- Model Cache → Model Checkpoint Loader: "Cached Model"
- Session Manager → AV Block Predictor: "Request Context"
- Error Handler → API Router: "Error Handling"

## INTERNAL ARROWS (24 Total)

### API Gateway (2 arrows)
1. Proxy Server → CORS Middleware: "Validate Origin"
2. CORS Middleware → API Router: "Approved Request"

### Request Processing (2 arrows)
3. File Upload Handler → Input Validator: "File Stream"
4. Input Validator → Temp Storage Manager: "Validated .npy"

### Signal Processing (2 arrows)
5. ECG Signal Loader → ECG Preprocessor: "Raw Array"
6. ECG Preprocessor → Signal Sampler: "Normalized Signal"

### ML Inference (7 arrows)
7. Model Checkpoint Loader → AV Block Predictor: "Loaded Weights" (dashed)
8. AV Block Predictor → U-NET Segmentation: "Batched Input"
9. U-NET Segmentation → Attention Module: "Feature Maps"
10. Attention Module → AV Block Classifier: "Enhanced Features"
11. AV Block Classifier → Wave Boundary Extractor: "Segmentation Masks"
12. Wave Boundary Extractor → U-NET Segmentation: "Wave Intervals"
13. U-NET Segmentation → Dissociation Detector: "PR/RR Intervals"

### XAI (3 arrows)
14. GRAD-CAM Generator → XAI Integrator: "Heatmap"
15. Saliency MAP generator → XAI Integrator: "Saliency Map"
16. XAI Integrator → Clinical Explain Generator: "Fused XAI Data"

### Diagnosis (2 arrows)
17. Clinical Rule Engine → Confidence Calculator: "AV Block Type"
18. Confidence Calculator → Severity Assessor: "Diagnosis + Confidence"

### Report Generation (3 arrows)
19. Clinical Report Builder → PDF Generator: "Aggregated Data"
20. PDF Generator → Report Storage Manager: "PDF Binary"
21. Report Storage Manager → Clinical Report Builder: "Report ID" (dashed)

### Response Formatting (3 arrows)
22. Data Serializer → Wave Interval Scaler: "JSON-safe Data"
23. Wave Interval Scaler → Response Builder: "Scaled Intervals"
24. Data Serializer → Response Builder: "Serialized Data"

## FORMATTING IN DRAW.IO

### High-Level Arrows:
1. Select arrow
2. Line width: 3pt
3. Color: Black
4. End arrow: Classic
5. Double-click to add label

### Internal Arrows:
1. Select arrow
2. Line width: 2pt
3. Color: Dark Gray (#666666)
4. End arrow: Classic
5. Double-click to add label

### Dashed Arrows (State Management):
1. Same as above
2. Line pattern: Dashed
3. Color: Light Gray (#999999)

## TOTAL: 41 Arrows
- High-level: 17
- Internal: 24

Good luck!
