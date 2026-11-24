# LungCare AI — Frontend Application

Modern clinical research workstation for chest radiograph classification and Grad-CAM visual explainability.

## 1. Technology Stack

- **Framework:** [React 18](https://react.dev/) with [TypeScript 5](https://www.typescriptlang.org/)
- **Build Tool:** [Vite 5](https://vitejs.dev/)
- **Styling:** [Tailwind CSS 3](https://tailwindcss.com/) with clinical teal/slate palette
- **Routing:** [React Router v6](https://reactrouter.com/)
- **Icons:** [Lucide React](https://lucide.dev/)
- **HTTP Client:** [Axios](https://axios-http.com/)

## 2. Directory Structure

```
frontend/
├── index.html                  # HTML entry point
├── package.json                # Dependencies and scripts
├── tsconfig.json               # TypeScript strict configuration
├── vite.config.ts              # Vite server & path aliases
├── tailwind.config.js          # Clinical design tokens
├── postcss.config.js           # PostCSS configuration
├── .env.example                # Template environment variables
│
└── src/
    ├── main.tsx                # Application bootstrap
    ├── App.tsx                 # Router & route definitions
    ├── index.css               # Tailwind directives & global styling
    │
    ├── api/                    # API client & endpoint definitions (Phase 3.3)
    ├── assets/                 # Static images and icons
    ├── components/
    │   ├── layout/             # Shell, Sidebar, Header, DisclaimerBanner
    │   └── ui/                 # Reusable Button, Card, Badge, Spinner
    ├── hooks/                  # Custom React hooks (Phase 3.3+)
    ├── pages/                  # 7 core application routes:
    │   ├── DashboardPage.tsx   # /dashboard
    │   ├── AnalysisPage.tsx    # /analysis
    │   ├── HistoryPage.tsx     # /history
    │   ├── ReportsPage.tsx     # /reports
    │   ├── ModelInfoPage.tsx   # /model-info
    │   ├── ProfilePage.tsx     # /profile
    │   └── SettingsPage.tsx    # /settings
    ├── types/                  # TypeScript interface contracts
    └── utils/                  # Styling & helper utilities (cn, storage)
```

## 3. Getting Started

### Prerequisites
- Node.js `v18+` or `v20+` (Tested on `v22.16.0`)
- npm `v9+` (Tested on `v11.6.1`)

### Installation & Development
```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start development server (runs on http://127.0.0.1:5173)
npm run dev

# Run TypeScript type check
npm run typecheck

# Build for production
npm run build
```

## 4. Application Routes
- `/dashboard`: System overview, classification categories, and quick actions.
- `/analysis`: Chest X-ray upload interface and interactive Grad-CAM heatmap overlay viewer.
- `/history`: Planned browser-local session history (Phase 3.7).
- `/reports`: Clinical research summary and printable PDF export.
- `/model-info`: DenseNet121 architecture specifications and validation metrics.
- `/profile`: User workspace preferences and local browser settings.
- `/settings`: API endpoint URL and visualization preference configuration.

## 5. Medical Safety Directive
> ⚠️ **Disclaimer:** LungCare AI is an experimental research prototype. Its predictions are not medical diagnoses and must not replace professional medical evaluation.
