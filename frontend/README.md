# FraudGuard - Frontend Dashboard

React + TypeScript dashboard for the Fraud Transaction Detection system, built with Vite and Tailwind CSS.

## Tech Stack

- **React 18** with TypeScript
- **Vite 6** for fast development and builds
- **Tailwind CSS** for styling
- **Recharts** for interactive charts
- **Lucide React** for icons

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Environment Variables

Create a `.env` file (already included):

```env
VITE_API_URL=http://localhost:8000
```

Update `VITE_API_URL` to point to your backend API if running on a different host/port.

## Features

- **Overview Tab**: Key metrics, best model stats, confusion matrix, dataset overview
- **Models Tab**: Model comparison chart, individual model cards with details
- **Analysis Tab**: Feature importance visualization, class imbalance analysis
- **Predict Tab**: Real-time fraud prediction form with risk analysis
- **Plots Tab**: Generated ML pipeline visualizations
