# Fetal Anomaly Detection System

A clinical decision-support web application for fetal ultrasound analysis. Combines a multi-head Vision Language Model (VLM) for image-based anomaly detection with a large language model (LLM) for maternal clinical risk assessment.

---

## Features

- **Plane Classification** — Automatically identifies the ultrasound plane (brain, thorax, NT, etc.)
- **Brain Anomaly Detection** — Classifies fetal brain scans into normal or anomalous categories (ventriculomegaly, holoprosencephaly, arachnoid cyst, etc.)
- **Nuchal Translucency Markers** — Measures NT thickness and nasal bone presence for Down syndrome screening
- **Heart Segmentation & CTR** — Segments cardiac structures and calculates the cardiothoracic ratio
- **LLM Clinical Risk Assessment** — Generates a structured 8-point clinical report from maternal vitals and history
- **Role-Based Access** — Separate views for doctors and system administrators

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19 + TypeScript (Vite) |
| Styling | TailwindCSS v4 |
| Database | Supabase (PostgreSQL) |
| VLM API | Deployed on HuggingFace Spaces (Gradio) |
| LLM | MBZUAI/MedMO-8B |

---

## Run Locally

**Prerequisites:** Node.js 18+

1. Clone the repository:
   ```bash
   git clone https://github.com/AliMusaRizvi/Fetal-Ultrasound.git
   cd Fetal-Ultrasound
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables — copy `.env.example` to `.env.local` and fill in your values:
   ```bash
   cp .env.example .env.local
   ```

4. Run the development server:
   ```bash
   npm run dev
   ```

The app runs on `http://localhost:3000` by default.

---

## Environment Variables

| Variable | Description |
|---|---|
| `VITE_SUPABASE_URL` | Your Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Supabase anonymous/publishable key |
| `VITE_VLM_API_URL` | VLM model API base URL |
| `VITE_LLM_API_URL` | LLM inference API endpoint |

---

## Project Structure

```
src/
├── lib/          # API clients (Supabase, VLM, LLM)
├── pages/
│   ├── dashboard/   # Dashboard pages (Upload, Reports, History, Doctors)
│   ├── Home.tsx
│   └── Login.tsx
└── components/   # Layout components
```

---


*Research prototype — for academic and demonstration purposes only. Not validated for clinical use.*
