import { TreePine, TrendingUp, AlertTriangle, Users, Droplets, Mountain } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, CartesianGrid,
  LineChart, Line, AreaChart, Area, Legend
} from 'recharts';

const COLORS = ['#22d3ee','#6366f1','#34d399','#fbbf24','#f87171','#a78bfa','#fb923c','#38bdf8','#e879f9','#4ade80'];

const chartTooltipStyle = {
  contentStyle: { background: '#1a2234', border: '1px solid #2a3650', borderRadius: 8, fontSize: '.82rem' },
  itemStyle: { color: '#94a3b8' },
  labelStyle: { color: '#f1f5f9', fontWeight: 600 },
};

export default function DashboardPage({ data }) {
  const { general_stats: s, by_species, by_location, by_monitoreo, distribution, feature_importance, model_metrics } = data;

  const topSpecies = by_species.slice(0, 10).map((d, i) => ({ ...d, fill: COLORS[i % COLORS.length] }));

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2>Dashboard de Análisis Ecológico</h2>
        <p>Resumen del análisis ML sobre el dataset de monitoreo cargado</p>
      </div>

      {/* Stat Cards */}
      <div className="stats-grid">
        <StatCard icon={<TreePine size={20}/>} color="cyan"
          value={s.n_trees} label="Árboles únicos" />
        <StatCard icon={<Users size={20}/>} color="green"
          value={s.n_species} label="Especies" />
        <StatCard icon={<TrendingUp size={20}/>} color="purple"
          value={`${(s.mean_growth * 100).toFixed(1)} cm`} label="Crecimiento promedio" />
        <StatCard icon={<AlertTriangle size={20}/>} color="amber"
          value={s.n_anomalies} label={`Anomalías (${s.pct_anomalies}%)`} />
        <StatCard icon={<Droplets size={20}/>} color="green"
          value={`${s.pct_zero_growth}%`} label="Sin crecimiento" />
        <StatCard icon={<Mountain size={20}/>} color="red"
          value={s.n_rows} label="Registros totales" />
      </div>

      {/* Charts Grid */}
      <div className="charts-grid">
        {/* Crecimiento por especie */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Crecimiento por Especie (Top 10)</span>
          </div>
          <ResponsiveContainer width="100%" height={270}>
            <BarChart data={topSpecies} layout="vertical" margin={{ left: 10, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3650" />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis type="category" dataKey="especie" width={140}
                     tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip {...chartTooltipStyle}
                formatter={(v) => [`${(v*100).toFixed(1)} cm`, 'Crecimiento']} />
              <Bar dataKey="mean_growth" radius={[0,4,4,0]}>
                {topSpecies.map((d, i) => <Cell key={i} fill={d.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Distribución del target */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Distribución de Crecimiento</span>
          </div>
          <ResponsiveContainer width="100%" height={270}>
            <AreaChart data={distribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3650" />
              <XAxis dataKey="bin" tick={{ fill: '#64748b', fontSize: 11 }}
                     tickFormatter={v => `${(v*100).toFixed(0)}`} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip {...chartTooltipStyle}
                labelFormatter={v => `${(v*100).toFixed(1)} cm`} />
              <Area type="monotone" dataKey="count" stroke="#22d3ee"
                    fill="rgba(34,211,238,.2)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Crecimiento por localidad */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Crecimiento por Localidad</span>
          </div>
          <ResponsiveContainer width="100%" height={270}>
            <BarChart data={by_location}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3650" />
              <XAxis dataKey="localidad" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip {...chartTooltipStyle}
                formatter={(v) => [`${(v*100).toFixed(1)} cm`, 'Crecimiento medio']} />
              <Bar dataKey="mean_growth" radius={[4,4,0,0]}>
                {by_location.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', marginTop: '.5rem' }}>
            {by_location.map((loc, i) => (
              <span key={i} style={{ fontSize: '.78rem', color: 'var(--text-muted)' }}>
                {loc.localidad}: <strong style={{ color: 'var(--text-secondary)' }}>{loc.n}</strong> registros
              </span>
            ))}
          </div>
        </div>

        {/* Evolución temporal */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Evolución por Monitoreo</span>
          </div>
          <ResponsiveContainer width="100%" height={270}>
            <LineChart data={by_monitoreo}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3650" />
              <XAxis dataKey="monitoreo" tick={{ fill: '#94a3b8', fontSize: 11 }}
                     tickFormatter={v => `Mon ${v}`} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip {...chartTooltipStyle}
                labelFormatter={v => `Monitoreo ${v}`}
                formatter={(v) => [`${(v*100).toFixed(1)} cm`]} />
              <Line type="monotone" dataKey="mean_growth" stroke="#a78bfa"
                    strokeWidth={2.5} dot={{ r: 5, fill: '#a78bfa', stroke: '#1a2234', strokeWidth: 2 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Feature Importance */}
        <div className="card" style={{ gridColumn: '1 / -1' }}>
          <div className="card-header">
            <span className="card-title">Importancia de Variables (GradientBoosting)</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '.45rem' }}>
            {(feature_importance || []).slice(0, 12).map((f, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '.8rem' }}>
                <span style={{ minWidth: 180, fontSize: '.82rem', color: 'var(--text-secondary)' }}>
                  {f.feature}
                </span>
                <div className="progress-bar-wrap" style={{ flex: 1 }}>
                  <div className="progress-bar-fill"
                       style={{
                         width: `${Math.max(f.importance * 100 / (feature_importance[0]?.importance || 1), 1)}%`,
                         background: i === 0 ? 'var(--accent)' : i < 3 ? '#6366f1' : '#334155'
                       }} />
                </div>
                <span style={{ minWidth: 55, textAlign: 'right', fontSize: '.82rem', fontWeight: 600,
                               color: i < 3 ? 'var(--accent)' : 'var(--text-muted)' }}>
                  {(f.importance * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, color, value, label }) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${color}`}>{icon}</div>
      <div>
        <div className="stat-value">{value}</div>
        <div className="stat-label">{label}</div>
      </div>
    </div>
  );
}
