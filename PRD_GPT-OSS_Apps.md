
# Product Requirements Document (PRD)  
**Title:** Mobile Apps with GPT-OSS Integration  
**Apps Covered:** RocketNotes AI, MapTaster  
**Version:** 1.0  
**Date:** 2025-08-14  

---

## 1. Purpose

Enable both RocketNotes AI and MapTaster to leverage **GPT-OSS** for enhanced user experience:  

- **RocketNotes AI:** Summarize, enhance, and provide smart suggestions for user notes triggered via NFC or manual input.  
- **MapTaster:** Generate insightful summaries of reviews, provide reasoning behind recommendations, and improve user decision-making.  

**Goal:** Start with **local development for testing**, then move to **cloud deployment** for production usage.  

---

## 2. Background

- Current apps operate **entirely on mobile devices**, with standard storage, UI, and basic logic.  
- GPT-OSS is **too large to run directly on mobile devices**, so a **backend inference server** is required.  
- User wants **local experimentation first** with GTX1660Ti + CPU + 128GB RAM, then cloud deployment for production.  

---

## 3. Features

### 3.1 RocketNotes AI

| Feature | Description | Implementation Notes |
|---------|-------------|--------------------|
| Note Summarization | Generate concise summaries of user notes | Mobile app sends text to GPT-OSS server → receives summary |
| Smart Suggestions | Generate tips, completions, or highlights | Use prompt templates to standardize outputs |
| NFC Integration | Use NFC input to trigger GPT-OSS enhancements | Mobile app reads NFC → sends text to backend API |
| Offline / Local Mode | Optional testing using local server | Limited batch size, only for development |

### 3.2 MapTaster

| Feature | Description | Implementation Notes |
|---------|-------------|--------------------|
| Review Summarization | Summarize user reviews into key insights | Backend GPT-OSS server receives review text → structured output |
| Recommendation Explanation | Provide reasons why a restaurant/bar is recommended | Use structured prompts to generate pros/cons summary |
| Filtering & Reliability | Highlight unreliable reviews, emphasizing trustworthy ones | GPT-OSS analyzes sentiment + text patterns |
| Batch Processing | Process multiple reviews in one request | Reduces inference overhead |

---

## 4. Architecture

```
Mobile App ⇄ Backend GPT-OSS Server ⇄ Model Inference
```

- **Mobile App:** UI, data collection, NFC reading, sending requests.  
- **Backend Server:**  
  - Runs GPT-OSS (local or cloud)  
  - Receives requests via REST API  
  - Returns text summaries or structured outputs  
- **Model Inference:**  
  - Initially local (GTX1660Ti + CPU + RAM)  
  - Later cloud GPU for production  
- **Optional Caching:** Store frequent outputs to reduce repeated inference  

---

## 5. Technical Requirements

### 5.1 Local Development
- OS: Windows 10 / Linux  
- GPU: GTX1660Ti  
- CPU: Core i7  
- RAM: 128GB  
- Storage: Enough for model weights (~several GB)  

### 5.2 Backend Server
- REST API (Flask / FastAPI)  
- Handles text input and returns GPT-OSS output  
- Optional: Batch processing, caching  

### 5.3 Mobile App
- Android / iOS  
- Handles API calls to backend  
- Displays GPT-generated summaries or suggestions  

### 5.4 Cloud Deployment (Future)
- Cloud GPU instance (AWS, GCP, Azure)  
- REST API endpoint accessible from mobile apps  
- Auto-scaling or on-demand usage to save costs  

---

## 6. Development Plan

1. **Phase 1 – Local Testing**
   - Install GPT-OSS locally  
   - Build REST API server  
   - Connect mobile apps for testing small batches  
   - Evaluate model output quality  

2. **Phase 2 – Prompt Optimization**
   - Create standardized prompts for summaries, suggestions, and review reasoning  
   - Tune outputs to match app UX requirements  

3. **Phase 3 – Cloud Deployment**
   - Deploy GPT-OSS on cloud GPU instance  
   - Connect mobile apps via secure API  
   - Implement caching and batch processing  

4. **Phase 4 – Production Release**
   - Full mobile integration  
   - Monitoring and scaling in cloud  
   - User feedback loop for improvement  

---

## 7. Constraints

- Mobile devices cannot run GPT-OSS locally  
- Local GPU memory may limit batch size  
- Cloud GPU costs must be managed for production  

---

## 8. KPIs / Success Metrics

- Average response time < 5 seconds per request (local testing)  
- Summarization accuracy and usefulness rated ≥ 4/5 by beta users  
- Mobile apps remain responsive and lightweight  
- Successful cloud deployment without downtime  

---

## 9. Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Model too large for local GPU | Reduce batch size, use smaller GPT-OSS variants if available |
| High inference latency | Use batching and caching |
| Cloud costs | Deploy on-demand instances, monitor usage |
| Integration complexity | Develop clear REST API interface for both apps |

---

## 10. Notes / Future Considerations

- Consider **fine-tuning GPT-OSS** for domain-specific outputs (notes, reviews)  
- Explore **serverless inference solutions** to minimize cloud costs  
- Extend features: multi-language support, advanced summarization, contextual suggestions  
