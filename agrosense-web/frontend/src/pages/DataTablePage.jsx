import { useState, useMemo } from 'react';
import { Search, ChevronDown } from 'lucide-react';

export default function DataTablePage({ data }) {
  const rows = data.table || [];
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const [sortCol, setSortCol] = useState(null);
  const [sortDir, setSortDir] = useState('asc');
  const perPage = 20;

  const columns = [
    { key: 'unique_tree_id', label: 'Árbol ID' },
    { key: 'monitoreo', label: 'Mon.' },
    { key: 'especie', label: 'Especie' },
    { key: 'localidad_str', label: 'Localidad' },
    { key: 'altura_anterior', label: 'Alt. Ant. (m)' },
    { key: 'altura_actual', label: 'Alt. Act. (m)' },
    { key: 'copa_anterior', label: 'Copa Ant. (m)' },
    { key: 'copa_actual', label: 'Copa Act. (m)' },
    { key: 'crecimiento_altura', label: 'Crec. (m)' },
    { key: 'pred_gb', label: 'Pred. GB (m)' },
    { key: 'is_anomaly', label: 'Anomalía' },
  ];

  const filtered = useMemo(() => {
    let d = rows;
    if (search.trim()) {
      const q = search.toLowerCase();
      d = d.filter(r =>
        Object.values(r).some(v =>
          String(v).toLowerCase().includes(q)
        )
      );
    }
    if (sortCol) {
      d = [...d].sort((a, b) => {
        let va = a[sortCol], vb = b[sortCol];
        if (typeof va === 'number' && typeof vb === 'number')
          return sortDir === 'asc' ? va - vb : vb - va;
        return sortDir === 'asc'
          ? String(va).localeCompare(String(vb))
          : String(vb).localeCompare(String(va));
      });
    }
    return d;
  }, [rows, search, sortCol, sortDir]);

  const totalPages = Math.ceil(filtered.length / perPage);
  const pageData = filtered.slice(page * perPage, (page + 1) * perPage);

  const handleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('asc'); }
    setPage(0);
  };

  const fmtVal = (key, val) => {
    if (key === 'is_anomaly') {
      return val === true || val === 'True' || val === 1
        ? <span className="badge anomaly">Sí</span>
        : <span className="badge normal">No</span>;
    }
    if (typeof val === 'number') return val.toFixed(4);
    return val ?? '-';
  };

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2>Datos Procesados</h2>
        <p>Dataset con predicciones y detección de anomalías ({rows.length} registros mostrados)</p>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
        <div className="search-wrap">
          <Search size={16} />
          <input
            className="search-input"
            placeholder="Buscar por especie, ID, localidad..."
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0); }}
          />
        </div>
        <span style={{ color: 'var(--text-muted)', fontSize: '.82rem', marginLeft: 'auto' }}>
          {filtered.length} resultados
        </span>
      </div>

      <div className="card" style={{ padding: 0 }}>
        <div className="table-wrap" style={{ maxHeight: 520, overflowY: 'auto' }}>
          <table>
            <thead>
              <tr>
                {columns.map(c => (
                  <th key={c.key} onClick={() => handleSort(c.key)}
                      style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}>
                    {c.label}
                    {sortCol === c.key && (
                      <ChevronDown size={12}
                        style={{ marginLeft: 3, verticalAlign: -2,
                                 transform: sortDir === 'desc' ? 'rotate(180deg)' : 'none',
                                 transition: 'transform .2s' }} />
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pageData.map((row, i) => (
                <tr key={i}>
                  {columns.map(c => (
                    <td key={c.key}>{fmtVal(c.key, row[c.key])}</td>
                  ))}
                </tr>
              ))}
              {pageData.length === 0 && (
                <tr>
                  <td colSpan={columns.length} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                    Sin resultados
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button disabled={page === 0} onClick={() => setPage(0)}>«</button>
          <button disabled={page === 0} onClick={() => setPage(p => p - 1)}>‹</button>
          {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
            let n;
            if (totalPages <= 7) n = i;
            else if (page < 4) n = i;
            else if (page > totalPages - 5) n = totalPages - 7 + i;
            else n = page - 3 + i;
            return (
              <button key={n} className={page === n ? 'active' : ''} onClick={() => setPage(n)}>
                {n + 1}
              </button>
            );
          })}
          <button disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}>›</button>
          <button disabled={page >= totalPages - 1} onClick={() => setPage(totalPages - 1)}>»</button>
          <span style={{ marginLeft: '.5rem' }}>({filtered.length} registros)</span>
        </div>
      )}
    </div>
  );
}
