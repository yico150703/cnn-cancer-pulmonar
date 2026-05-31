require('dotenv').config();
const express = require('express');
const cors    = require('cors');
const multer  = require('multer');
const path    = require('path');
const fs      = require('fs');
const { predecir, cargarModelo } = require('./predecir');

const app  = express();
const PORT = process.env.PORT || 3001;

// ── CORS completamente abierto ────────────────────────────────────────────────
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

app.use(cors());
app.use(express.json());

// ── Multer ────────────────────────────────────────────────────────────────────
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
app.get('/', (_req, res) => {
  res.json({
    status:  'online',
    api:     'CNN Cáncer Pulmonar MRI',
    version: '1.0.0',
    rutas:   ['GET /', 'GET /metricas', 'POST /predecir']
  });
});

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

app.post('/predecir', upload.single('imagen'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No se recibió ninguna imagen.' });
    }
    console.log(`📷 ${req.file.originalname}`);
    const resultado = await predecir(req.file.buffer);
    console.log(`→ ${resultado.clase} | ${resultado.confianza}%`);
    res.json({ exito: true, archivo: req.file.originalname, resultado });
  } catch (error) {
    console.error('❌', error.message);
    res.status(500).json({ error: error.message });
  }
});

// ── Iniciar ───────────────────────────────────────────────────────────────────
app.listen(PORT, async () => {
  console.log(`\n🚀 API en http://localhost:${PORT}`);
  try {
    await cargarModelo();
  } catch {
    console.warn('⚠️  Modelo no disponible. Ejecuta: npm run train\n');
  }
});
