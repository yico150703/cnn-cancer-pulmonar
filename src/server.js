// src/server.js
// API REST — expone el modelo CNN para predicción vía HTTP

require('dotenv').config();
const express = require('express');
const cors    = require('cors');
const multer  = require('multer');
const path    = require('path');
const fs      = require('fs');
const { predecir, cargarModelo } = require('./predecir');

const app  = express();
const PORT = process.env.PORT || 3001;

// ── Middlewares ───────────────────────────────────────────────────────────────
app.use(cors({
  origin:  process.env.FRONTEND_URL || '*',
  methods: ['GET', 'POST']
}));
app.use(express.json());

// ── Multer: archivos en memoria ───────────────────────────────────────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits:  { fileSize: 20 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (/\.(jpg|jpeg|png|bmp)$/i.test(file.originalname)) {
      cb(null, true);
    } else {
      cb(new Error('Solo se aceptan imágenes JPG, PNG o BMP'));
    }
  }
});

// ── Rutas ─────────────────────────────────────────────────────────────────────

// Health check
app.get('/', (_req, res) => {
  res.json({
    status:    'online',
    api:       'CNN Cáncer Pulmonar MRI',
    version:   '1.0.0',
    clases:    ['cancer', 'no_cancer'],
    threshold: 0.5,
    nota:      'sigmoid >= 0.5 → cancer'
  });
});

// Métricas de entrenamiento (para mostrar en el frontend)
app.get('/metricas', (_req, res) => {
  const metaPath = path.join(__dirname, '..', 'modelo_guardado', 'meta.json');
  if (!fs.existsSync(metaPath)) {
    return res.status(404).json({ error: 'Modelo no entrenado aún' });
  }
  const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  res.json({
    historial:   meta.historial,
    conteoTrain: meta.conteoTrain,
    conteoVal:   meta.conteoVal
  });
});

// Predicción
app.post('/predecir', upload.single('imagen'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No se recibió ninguna imagen.' });
    }

    console.log(`📷 ${req.file.originalname} (${(req.file.size/1024).toFixed(1)} KB)`);

    const resultado = await predecir(req.file.buffer);

    console.log(
      `   → ${resultado.clase} | confianza: ${resultado.confianza}% | sigmoid: ${resultado.valorSigmoid}`
    );

    res.json({ exito: true, archivo: req.file.originalname, resultado });

  } catch (error) {
    console.error('❌', error.message);
    res.status(500).json({ error: error.message });
  }
});

// ── Iniciar servidor ──────────────────────────────────────────────────────────
app.listen(PORT, async () => {
  console.log(`\n🚀 API corriendo en http://localhost:${PORT}`);
  try {
    await cargarModelo();
  } catch {
    console.warn('⚠️  Modelo no disponible. Ejecuta: npm run train\n');
  }
});