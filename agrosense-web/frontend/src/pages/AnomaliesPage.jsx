import { useState } from 'react';
import { AlertTriangle, TrendingUp, TreePine } from 'lucide-react';
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Cell, BarChart, Bar
} from 'recharts';

const COLORS = ['#22d3ee','#6366f1','#34d399','#fbbf24','#f87171','#a78bfa'];

export default function AnomaliesPage({ data }) {
  const { anomalies, general_stats: s } = data;
  const [page, setPage] = useState(0);
  const perPage = 15;
  const totalPages = Math.ceil(anomalies.length / perPage);
  const pageData = anomalies.slice(page * perPage, (page + 1) * perPage);

  // Anomalías por especie
  const speciesCount = {};
  anomalies.forEach(a => { speciesCount[a.especie] = (speciesCount[a.especie] || 0) + 1; });
  const bySpecies = Object.entries(speciesCount)
    .map(([especie, count]) => ({ especie, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);

  // Scatter data
  const scatterData = anomalies.map(a => ({
    x: a.altura_actual || 0,
    y: a.crecimiento_altura || 0,
    name: a.unique_tree_id,
    score: a.anomaly_score,
  }));

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2>Detección de Anomalías</h2>
        <p>Árboles con comportamiento atípico identificados por Isolation Forest</p>
      </div>

      <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <div className="stat-card">
          <div className="stat-icon amber"><AlertTriangle size={20} /></div>
          <div>
            <div className="stat-value">{s.n_anomalies}</div>
            <div className="stat-label">Anomalías detectadas</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon purple"><TrendingUp size={20} /></div>
          <div>
            <div className="stat-value">{s.pct_anomalies}%</div>
            <div className="stat-label">Tasa de anomalía</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon green"><TreePine size={20} /></div>
          <div>
            <div className="stat-value">{bySpecies[0]?.especie?.split(' ')[0] || '-'}</div>
            <div className="stat-label">Especie más anómala</div>
          </div>
        </div>
      </div>

      <div className="charts-grid" style={{ marginBottom: '1.5rem' }}>
        {/* Scatter anomalías */}
        <div className="card">
          <div className="card-header"><span className="card-title">Anomalías: Altura vs Crecimiento</span></div>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 25, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3650" />
              <XAxis dataKey="x" name="Altura actual" unit=" m" type="number"
                     tick={{ fill: '#64748b', fontSize: 11 }}
                     label={{ value: 'Altura Actual (m)', position: 'insideBottom', offset: -15, fill: '#94a3b8', fontSize: 12 }} />
              <YAxis dataKey="y" name="Crecimiento" unit=" m" type="number"
                     tick={{ fill: '#64748b', fontSize: 11 }}
                     label={{ value: 'Crecimiento (m)', angle: -90, position: 'insideLeft', offset: -5, fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip
                contentStyle={{ background: '#1a2234', border: '1px solid #2a3650', borderRadius: 8, fontSize: '.82rem' }}
                formatter={(v, n) => [typeof v === 'number' ? `${v.toFixed(3)} m` : v, n]}
              />
              <Scatter data={scatterData} fill="#fbbf24">
                {scatterData.map((_, i) => (
                  <Cell key={i} fill="#fbbf24" opacity={0.8} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Anomalías por especie */}
        <div className="card">
          <div className="card-header"><span className="card-title">Anomalías por Especie</span></div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={bySpecies} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3650" />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis type="category" dataKey="especie" width={130}
                     tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#1a2234', border: '1px solid #2a3650', borderRadius: 8, fontSize: '.82rem' }}
              />
              <Bar dataKey="count" radius={[0,4,4,0]}>
                {bySpecies.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Interpretación */}
      <div className="card" style={{ marginBottom: '1.5rem', borderLeft: '3px solid var(--amber)' }}>
        <div className="card-title" style={{ marginBottom: '.5rem' }}>
          <AlertTriangle size={16} style={{ verticalAlign: -3, marginRight: 6, color: 'var(--amber)' }} />
          Interpretación Ecológica
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '.88rem', lineHeight: 1.7 }}>
          Las anomalías detectadas <strong>no son errores</strong>: corresponden a árboles de alto desempeño,
          principalmente especies pioneras de crecimiento rápido como <em>{bySpecies[0]?.especie || 'N/A'}</em>.
          Estos individuos presentan tasas de crecimiento significativamente superiores al promedio,
          lo que los convierte en candidatos para estudios de caso de éxito en restauración ecológica.
        </p>
      </div>

      {/* Tabla de anomalías */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Registros Anómalos ({anomalies.length})</span>
        </div>
        <div className="table-wrap" style={{ maxHeight: 450, overflowY: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Árbol ID</th>
                <th>Mon.</th>
                <th>Especie</th>
                <th>Crec. Altura</th>
                <th>Alt. Anterior</th>
                <th>Alt. Actual</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {pageData.map((row, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 500 }}>{row.unique_tree_id}</td>
                  <td>{row.monitoreo}</td>
                  <td><em>{row.especie}</em></td>
                  <td style={{ color: 'var(--amber)', fontWeight: 600 }}>
                    {typeof row.crecimiento_altura === 'number' ? `${(row.crecimiento_altura * 100).toFixed(1)} cm` : '-'}
                  </td>
                  <td>{typeof row.altura_anterior === 'number' ? `${row.altura_anterior.toFixed(2)} m` : '-'}</td>
                  <td>{typeof row.altura_actual === 'number' ? `${row.altura_actual.toFixed(2)} m` : '-'}</td>
                  <td style={{ color: 'var(--red)', fontSize: '.8rem' }}>
                    {typeof row.anomaly_score === 'number' ? row.anomaly_score.toFixed(4) : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {totalPages > 1 && (
          <div className="pagination">
            <button disabled={page === 0} onClick={() => setPage(p => p - 1)}>Anterior</button>
            <span>Página {page + 1} de {totalPages}</span>
            <button disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}>Siguiente</button>
          </div>
        )}
      </div>
    </div>
  );
}
