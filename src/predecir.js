// src/predecir.js
// Carga el modelo guardado y clasifica nuevas imágenes MRI.
//
// Equivale a la sección "Confusion Matrix" del notebook:
//   preds = model.predict(x_batch)
//   y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
//   < 0.5  → no_cancer
//   >= 0.5 → cancer

const tf   = require('@tensorflow/tfjs');
const path = require('path');
const fs   = require('fs');
const Jimp = require('jimp');

const SAVE_DIR = path.join(__dirname, '..', 'modelo_guardado');

let modeloCargado = null;
let meta          = null;

// Custom IOHandler to load model from a folder without tfjs-node
function createLoadHandlerFromDir(dir) {
  return {
    load: async () => {
      const modelJsonPath = path.join(dir, 'model.json');
      if (!fs.existsSync(modelJsonPath)) {
        throw new Error('model.json no encontrado en ' + dir);
      }
      const modelJSON = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));

      // Assume single weights file as produced by the saver below
      const manifest = modelJSON.weightsManifest && modelJSON.weightsManifest[0];
      const weightPath = manifest && manifest.paths && manifest.paths[0];
      if (!weightPath) throw new Error('weights file not found in manifest');

      const weightBuffer = fs.readFileSync(path.join(dir, weightPath));
      // Convert Node Buffer to ArrayBuffer
      const arrayBuffer = weightBuffer.buffer.slice(weightBuffer.byteOffset, weightBuffer.byteOffset + weightBuffer.byteLength);

      return {
        modelTopology: modelJSON.modelTopology,
        weightSpecs: manifest.weights,
        weightData: arrayBuffer,
        format: modelJSON.format || 'layers-model',
        generatedBy: modelJSON.generatedBy,
        convertedBy: modelJSON.convertedBy
      };
    }
  };
}

// ── Carga perezosa del modelo ─────────────────────────────────────────────────
async function cargarModelo() {
  if (modeloCargado) return;

  const metaPath = path.join(SAVE_DIR, 'meta.json');
  if (!fs.existsSync(metaPath)) {
    throw new Error('Modelo no encontrado. Ejecuta primero: npm run train');
  }

  meta          = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  // Use custom IOHandler to load without tfjs-node
  modeloCargado = await tf.loadLayersModel(createLoadHandlerFromDir(SAVE_DIR));

  console.log(`✅ Modelo cargado | imgSize: ${meta.imgSize}px | threshold: ${meta.threshold}`);
}

// ── Preprocesamiento idéntico al notebook ─────────────────────────────────────
// ImageDataGenerator(rescale=1./255) + color_mode='grayscale'
async function preprocesar(buffer) {
  const imgSize = meta?.imgSize || 224;
  const img     = await Jimp.read(buffer);
  img.resize(imgSize, imgSize).grayscale();

  const pixeles = new Float32Array(imgSize * imgSize * 1);
  let i = 0;
  img.scan(0, 0, img.bitmap.width, img.bitmap.height, (x, y, idx) => {
    pixeles[i++] = img.bitmap.data[idx] / 255.0; // rescale=1./255
  });

  return tf.tensor4d(pixeles, [1, imgSize, imgSize, 1]);
}

// ── Predicción ────────────────────────────────────────────────────────────────
async function predecir(buffer) {
  await cargarModelo();

  const tensor    = await preprocesar(buffer);
  const prediccion = modeloCargado.predict(tensor);
  const probCancer = (await prediccion.data())[0]; // valor sigmoid [0,1]

  tensor.dispose();
  prediccion.dispose();

  // Umbral 0.5 — equivale a: y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
  const threshold   = meta?.threshold || 0.5;
  const esCancer    = probCancer >= threshold;
  const clase       = esCancer ? 'cancer' : 'no_cancer';
  const confianza   = parseFloat(
    (esCancer ? probCancer : 1 - probCancer) * 100
  ).toFixed(2);

  return {
    clase,
    confianza:      parseFloat(confianza),
    probabilidades: {
      cancer:    parseFloat((probCancer * 100).toFixed(2)),
      no_cancer: parseFloat(((1 - probCancer) * 100).toFixed(2))
    },
    valorSigmoid: parseFloat(probCancer.toFixed(4))
  };
}

module.exports = { predecir, cargarModelo };