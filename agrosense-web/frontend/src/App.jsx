import { useState } from 'react';
import './index.css';
import {
  UploadCloud, BarChart3, AlertTriangle, Table2, Brain,
  TreePine, Activity
} from 'lucide-react';
import UploadPage from './pages/UploadPage';
import DashboardPage from './pages/DashboardPage';
import AnomaliesPage from './pages/AnomaliesPage';
import DataTablePage from './pages/DataTablePage';
import ModelInfoPage from './pages/ModelInfoPage';

const NAV = [
  { id: 'upload',    label: 'Cargar Datos',   icon: UploadCloud },
  { id: 'dashboard', label: 'Dashboard',      icon: BarChart3 },
  { id: 'anomalies', label: 'Anomalías',      icon: AlertTriangle },
  { id: 'table',     label: 'Datos',          icon: Table2 },
  { id: 'model',     label: 'Modelo ML',      icon: Brain },
];

export default function App() {
  const [page, setPage] = useState('upload');
  const [results, setResults] = useState(null);

  const handleResults = (data) => {
    setResults(data);
    setPage('dashboard');
  };

  const renderPage = () => {
    switch (page) {
      case 'upload':    return <UploadPage onResults={handleResults} />;
      case 'dashboard': return results ? <DashboardPage data={results} /> : <EmptyState onGo={() => setPage('upload')} />;
      case 'anomalies': return results ? <AnomaliesPage data={results} /> : <EmptyState onGo={() => setPage('upload')} />;
      case 'table':     return results ? <DataTablePage data={results} /> : <EmptyState onGo={() => setPage('upload')} />;
      case 'model':     return <ModelInfoPage data={results} />;
      default:          return null;
    }
  };

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon"><TreePine size={20} /></div>
          <h1>Agro<span>Sense</span> AI</h1>
        </div>
        <nav className="sidebar-nav">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${page === id ? 'active' : ''}`}
              onClick={() => setPage(id)}
            >
              <Icon size={18} />
              {label}
            </button>
          ))}
        </nav>
        <div style={{ padding: '1rem', borderTop: '1px solid var(--border-color)', marginTop: 'auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem' }}>
            <Activity size={14} style={{ color: results ? 'var(--green)' : 'var(--text-muted)' }} />
            <span style={{ fontSize: '.78rem', color: results ? 'var(--green)' : 'var(--text-muted)' }}>
              {results ? 'Dataset cargado' : 'Sin datos'}
            </span>
          </div>
        </div>
      </aside>
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  );
}

function EmptyState({ onGo }) {
  return (
    <div className="fade-in" style={{ textAlign: 'center', paddingTop: '10vh' }}>
      <UploadCloud size={56} style={{ color: 'var(--text-muted)', margin: '0 auto 1rem' }} />
      <h3 style={{ color: 'var(--text-secondary)', marginBottom: '.5rem' }}>No hay datos cargados</h3>
      <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
        Primero sube un archivo de monitoreo ecológico para ver resultados.
      </p>
      <button onClick={onGo} className="nav-item active" style={{ display: 'inline-flex', width: 'auto' }}>
        <UploadCloud size={16} /> Ir a cargar datos
      </button>
    </div>
  );
}
