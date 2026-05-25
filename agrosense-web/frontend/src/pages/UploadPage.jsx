import { useState, useRef } from 'react';
import { UploadCloud, FileSpreadsheet, CheckCircle, XCircle, Loader } from 'lucide-react';
import { uploadFile } from '../api';

export default function UploadPage({ onResults }) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState('');
  const inputRef = useRef();

  const handleFile = async (file) => {
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['xlsx', 'csv'].includes(ext)) {
      setError('Solo se aceptan archivos .xlsx o .csv');
      return;
    }
    setFileName(file.name);
    setError(null);
    setLoading(true);
    try {
      const res = await uploadFile(file);
      if (res.data.valid) {
        onResults(res.data.data);
      } else {
        setError(res.data.errors?.join('. ') || 'Error de validación');
      }
    } catch (e) {
      const msg = e.response?.data?.detail || e.response?.data?.errors?.join('. ') || e.message;
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); };
  const onDrag = (e) => { e.preventDefault(); setDragging(true); };
  const onLeave = (e) => { e.preventDefault(); setDragging(false); };

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2>Cargar Dataset de Monitoreo</h2>
        <p>Sube un archivo Excel (.xlsx) o CSV con los datos de monitoreo ecológico.</p>
      </div>

      <div className="card" style={{ maxWidth: 640, margin: '0 auto' }}>
        <div
          className={`upload-zone ${dragging ? 'dragging' : ''}`}
          onDrop={onDrop}
          onDragOver={onDrag}
          onDragLeave={onLeave}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef} type="file" hidden
            accept=".xlsx,.csv"
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {loading ? (
            <>
              <Loader size={48} style={{ color: 'var(--accent)' }} className="loading-pulse" />
              <p style={{ marginTop: '.75rem' }}>
                Procesando <strong>{fileName}</strong>...
              </p>
              <p>Ejecutando pipeline ML, esto puede tomar unos segundos.</p>
            </>
          ) : (
            <>
              <UploadCloud size={48} style={{ color: 'var(--accent)' }} />
              <p><span className="accent-text">Haz clic para seleccionar</span> o arrastra un archivo aquí</p>
              <p style={{ fontSize: '.8rem', marginTop: '.5rem' }}>Formatos aceptados: .xlsx, .csv</p>
            </>
          )}
        </div>

        {error && (
          <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--red-dim)',
                        borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'flex-start', gap: '.7rem' }}>
            <XCircle size={18} style={{ color: 'var(--red)', flexShrink: 0, marginTop: 2 }} />
            <div>
              <strong style={{ color: 'var(--red)' }}>Error de validación</strong>
              <p style={{ color: 'var(--text-secondary)', fontSize: '.85rem', marginTop: 4 }}>{error}</p>
            </div>
          </div>
        )}
      </div>

      <div className="card" style={{ maxWidth: 640, margin: '1.5rem auto 0' }}>
        <div className="card-header">
          <span className="card-title">
            <FileSpreadsheet size={16} style={{ marginRight: 6, verticalAlign: -3 }} />
            Columnas requeridas
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '.3rem .5rem', fontSize: '.83rem', color: 'var(--text-secondary)' }}>
          {[
            'unique_tree_id', 'monitoreo', 'especie', 'familia',
            'nombre_comun', 'meses_desde_inicio', 'elevacion_m',
            'coord_x', 'coord_y', 'altura_anterior', 'copa_anterior',
            'crecimiento_altura', 'crecimiento_copa',
            'fito_Bueno', 'fito_Malo', 'fito_Regular',
            'gremio_Inicial', 'gremio_Intermedia',
            'loc_Guayabal', 'loc_San Antonio', 'loc_Tres Jotas'
          ].map(c => (
            <div key={c} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <CheckCircle size={12} style={{ color: 'var(--green)' }} />
              <code style={{ fontSize: '.78rem' }}>{c}</code>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
