import { useState, useEffect } from 'react';
import { Brain, Award, Target, Gauge } from 'lucide-react';
import { getModelInfo } from '../api';

export default function ModelInfoPage({ data }) {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModelInfo()
      .then(res => setInfo(res.data))
      .catch(() => setInfo(null))
      .finally(() => setLoading(false));
  }, []);

  // Preferir datos del pipeline si ya existen, fallback a /model-info
  const metrics = data?.model_metrics || info?.metrics;
  const fi = data?.feature_importance || info?.feature_importance || [];
  const stats = info?.dataset_stats;

  if (loading) {
    return (
      <div className="fade-in" style={{ textAlign: 'center', paddingTop: '10vh' }}>
        <div className="loading-pulse" style={{ color: 'var(--text-muted)' }}>Cargando información del modelo...</div>
      </div>
    );
  }

  const gb = metrics?.GradientBoosting || {};
  const rf = metrics?.RandomForest || {};

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2>Modelo de Machine Learning</h2>
        <p>Métricas de evaluación, interpretación y detalles técnicos</p>
      </div>

      {/* Métricas principales GradientBoosting */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <span className="card-title">
            <Brain size={16} style={{ verticalAlign: -3, marginRight: 6 }} />
            GradientBoosting Regressor — Modelo Principal
          </span>
          <span className="badge normal">Mejor modelo</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
          <MetricCard
            icon={<Target size={20} />} color="cyan"
            name="R²" value={gb.r2?.toFixed(4) || '-'}
            desc={`Explica ${((gb.r2 || 0) * 100).toFixed(1)}% de la varianza`}
          />
          <MetricCard
            icon={<Gauge size={20} />} color="green"
            name="MAE" value={gb.mae ? `${(gb.mae * 100).toFixed(1)} cm` : '-'}
            desc="Error absoluto medio"
          />
          <MetricCard
            icon={<Award size={20} />} color="purple"
            name="RMSE" value={gb.rmse ? `${(gb.rmse * 100).toFixed(1)} cm` : '-'}
            desc="Raíz del error cuadrático medio"
          />
        </div>

        <div style={{ padding: '1rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', fontSize: '.88rem', lineHeight: 1.7 }}>
          <strong style={{ color: 'var(--accent)' }}>Interpretación:</strong>
          <span style={{ color: 'var(--text-secondary)' }}>
            {' '}El modelo predice el crecimiento en altura con un error promedio de{' '}
            <strong>{gb.mae ? `${(gb.mae * 100).toFixed(1)} cm` : '?'}</strong>.
            {gb.r2 >= 0.95 && ' Este nivel de precisión es excelente para datos ecológicos de campo.'}
            {gb.r2 >= 0.90 && gb.r2 < 0.95 && ' El rendimiento es muy bueno para datos de campo con variabilidad natural.'}
            {gb.r2 < 0.90 && ' El rendimiento es aceptable, pero podría mejorarse con más variables o datos.'}
          </span>
        </div>
      </div>

      {/* Comparación de modelos */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <span className="card-title">Comparación de Modelos</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Modelo</th><th>R²</th><th>MAE (cm)</th><th>RMSE (cm)</th><th>Resultado</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ background: 'var(--accent-dim)' }}>
                <td><strong>GradientBoosting</strong></td>
                <td style={{ color: 'var(--accent)', fontWeight: 700 }}>{gb.r2?.toFixed(4)}</td>
                <td>{gb.mae ? (gb.mae * 100).toFixed(1) : '-'}</td>
                <td>{gb.rmse ? (gb.rmse * 100).toFixed(1) : '-'}</td>
                <td><span className="badge normal">Mejor</span></td>
              </tr>
              <tr>
                <td>RandomForest</td>
                <td>{rf.r2?.toFixed(4)}</td>
                <td>{rf.mae ? (rf.mae * 100).toFixed(1) : '-'}</td>
                <td>{rf.rmse ? (rf.rmse * 100).toFixed(1) : '-'}</td>
                <td><span style={{ color: 'var(--text-muted)', fontSize: '.8rem' }}>Alternativo</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <span className="card-title">Top Variables Predictivas</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
          {fi.slice(0, 10).map((f, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '.8rem' }}>
              <span style={{ minWidth: 22, textAlign: 'right', fontSize: '.82rem', fontWeight: 700, color: 'var(--text-muted)' }}>
                {i + 1}.
              </span>
              <span style={{ minWidth: 180, fontSize: '.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
                {f.feature}
              </span>
              <div className="progress-bar-wrap" style={{ flex: 1 }}>
                <div className="progress-bar-fill"
                     style={{
                       width: `${Math.max(f.importance * 100 / (fi[0]?.importance || 1), 1)}%`,
                       background: i === 0 ? 'var(--accent)' : i < 3 ? '#6366f1' : '#475569'
                     }} />
              </div>
              <span style={{ minWidth: 55, textAlign: 'right', fontSize: '.82rem', fontWeight: 600,
                             color: i < 3 ? 'var(--accent)' : 'var(--text-muted)' }}>
                {(f.importance * 100).toFixed(2)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Detalles técnicos */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Detalles Técnicos</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', fontSize: '.88rem' }}>
          <Detail label="Algoritmo" value="GradientBoostingRegressor" />
          <Detail label="n_estimators" value="300" />
          <Detail label="learning_rate" value="0.05" />
          <Detail label="max_depth" value="5" />
          <Detail label="Anomalías" value="Isolation Forest (5% contamination)" />
          <Detail label="Split" value="80% train / 20% test" />
          {stats && (
            <>
              <Detail label="Dataset entrenamiento" value={`${stats.n_rows} registros`} />
              <Detail label="Especies" value={stats.n_species} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricCard({ icon, color, name, value, desc }) {
  return (
    <div style={{ padding: '1.2rem', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)',
                  borderLeft: `3px solid var(--${color})` }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '.5rem', marginBottom: '.5rem' }}>
        <span style={{ color: `var(--${color})` }}>{icon}</span>
        <span style={{ fontWeight: 600, color: 'var(--text-muted)', fontSize: '.82rem' }}>{name}</span>
      </div>
      <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{value}</div>
      <div style={{ fontSize: '.78rem', color: 'var(--text-muted)', marginTop: 4 }}>{desc}</div>
    </div>
  );
}

function Detail({ label, value }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '.5rem .75rem',
                  background: 'var(--bg-secondary)', borderRadius: 6 }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>{value}</span>
    </div>
  );
}
