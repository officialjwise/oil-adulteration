# Oil Adulteration Detection API Integration Guide

This document provides a comprehensive, non-coding integration reference for frontend developers. It describes all available endpoints, request formats, authentication flows, headers, and sample responses.

---

## Base URL

> http://localhost:8000

_All endpoints are relative to this base URL._

---

## Authentication

### 1. Sign Up (Demo or Firebase)

- **Endpoint:** `POST /auth/signup`
- **Authentication:** Not required
- **Request Body (JSON):**
  ```json
  {
    "name": "Your Name",
    "email": "user@example.com",
    "password": "securePassword123"
  }
  ```
- **Response (200):**
  ```json
  {
    "token": "<JWT or demo token>",
    "user_id": "<UUID or Firebase UID>",
    "email": "user@example.com",
    "name": "Your Name"
  }
  ```

### 2. Sign In (Demo)

- **Endpoint:** `POST /auth/signin`
- **Authentication:** Not required
- **Request Body (JSON):**
  ```json
  {
    "email": "user@example.com",
    "password": "securePassword123"
  }
  ```
- **Response (200):**
  ```json
  {
    "token": "<JWT or demo token>",
    "user_id": "<UUID or Firebase UID>",
    "email": "user@example.com",
    "name": "Your Name"
  }
  ```

### 3. Firebase Custom-Token Flow (Production)

1. Generate a custom token in your backend (Firebase Admin SDK).
2. On the client, exchange the custom token for an ID token:
   ```bash
   POST https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key=<API_KEY>
   Content-Type: application/json
   {
     "token": "<CUSTOM_TOKEN>",
     "returnSecureToken": true
   }
   ```
3. Extract `idToken` from the JSON response.
4. Use this `idToken` as the Bearer token in protected endpoints.

---

## Common Header for Protected Routes

```http
Authorization: Bearer <ID_TOKEN>
Content-Type: application/json
```

> If you see `401 Unauthorized`, refresh or re-obtain the token.

---

## Endpoints Reference

### 1. Health Check

- **Endpoint:** `GET /health`
- **Auth:** No
- **Response (200):**
  ```json
  {
    "status": "healthy",
    "timestamp": "2025-08-08T05:11:06.488919",
    "model_loaded": true,
    "firebase_enabled": false
  }
  ```

### 2. Root / Info

- **Endpoint:** `GET /`
- **Auth:** No
- **Response (200):**
  ```json
  {
    "message": "Oil Adulteration Detection API is running.",
    "version": "1.0.0",
    "supported_oils": ["Palm Oil", "Groundnut Oil"],
    "endpoints": {
      "auth": ["/auth/signup", "/auth/signin"],
      "analysis": ["/predict/"],
      "analytics": ["/analytics/recent", "/analytics/summary"]
    }
  }
  ```

### 3. Predict Analysis

- **Endpoint:** `POST /predict/`
- **Auth:** Yes
- **Content Type:** `multipart/form-data`
- **Form Field:** `file` &mdash; CSV file upload
- **Sample cURL:**
  ```bash
  curl -X POST "http://localhost:8000/predict/" \
    -H "Authorization: Bearer <ID_TOKEN>" \
    -F "file=@/path/to/your.csv"
  ```
- **Response (200):**
  ```json
  {
    "results": [
      {
        "sample_id": "SAMPLE-001",
        "oil_type": "Palm Oil",
        "status": "Pure",
        "confidence": 0.95,
        "analysis_time": "47s",
        "timestamp": "2025-08-08T05:10:45.043208",
        "user_id": "<UID>"
      },
      ...
    ]
  }
  ```
- **Error (400/500):** JSON with `detail` field.

### 4. Recent Analytics

- **Endpoint:** `GET /analytics/recent`
- **Auth:** Yes
- **Query Parameters:**
  - `days` (optional, default `7`) &mdash; lookback window
  - `oil_type` (optional) &mdash; `Palm Oil` or `Groundnut Oil`
  - `status` (optional) &mdash; `Pure` or `Adulterated`
- **Sample cURL:**
  ```bash
  curl -X GET "http://localhost:8000/analytics/recent?days=30&status=Pure" \
    -H "Authorization: Bearer <ID_TOKEN>"
  ```
- **Response (200):**
  ```json
  {
    "total_analyses": 5,
    "pure_count": 4,
    "adulterated_count": 1,
    "palm_oil_count": 3,
    "groundnut_oil_count": 2,
    "analyses": [ /* array of AnalysisRecord */ ]
  }
  ```

### 5. Analytics Summary

- **Endpoint:** `GET /analytics/summary`
- **Auth:** Yes
- **Response (200):**
  ```json
  {
    "total_analyses": 10,
    "recent_analyses": 3,
    "purity_rate": 80.0,
    "most_analyzed_oil": "Palm Oil"
  }
  ```

---

## Data Models (Frontend TypeScript Interfaces)
```ts
interface SignUpRequest {
  name: string;
  email: string;
  password: string;
}

interface AuthResponse {
  token: string;
  user_id: string;
  email: string;
  name: string;
}

interface AnalysisRecord {
  sample_id: string;
  oil_type: 'Palm Oil' | 'Groundnut Oil';
  status: 'Pure' | 'Adulterated';
  confidence: number;
  analysis_time: string;
  timestamp: string; // ISO
  user_id?: string;
}

interface AnalyticsResponse {
  total_analyses: number;
  pure_count: number;
  adulterated_count: number;
  palm_oil_count: number;
  groundnut_oil_count: number;
  analyses: AnalysisRecord[];
}

interface SummaryResponse {
  total_analyses: number;
  recent_analyses: number;
  purity_rate: number;
  most_analyzed_oil: string;
}
```  

---

### Tips & Best Practices

- Ensure `Authorization` header is refreshed before expiry.
- Handle HTTP status codes: `401` &rarr; redirect to login; `400`/`500` &rarr; show error messages.
- Use `multipart/form-data` for file uploads; many HTTP client libraries (Axios, Fetch) support `FormData`.
- Parse JSON responses and map to TypeScript interfaces for type safety.

---

_This guide is auto-generated. Reach out to the backend team for any changes or updates._
