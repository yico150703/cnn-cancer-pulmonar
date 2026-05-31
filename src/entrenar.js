// src/entrenar.js
// Entrenamiento de la CNN para detección de cáncer pulmonar en MRI
// Referencia: kaggle.com/code/truenhatnam/cnn-lungcancer
//
// Equivalentes Python → JavaScript:
//   ImageDataGenerator        → carga manual con Jimp + augmentation
//   class_weight='balanced'   → compute_class_weight manual
//   EarlyStopping             → lógica manual con paciencia
//   ReduceLROnPlateau         → reducción de LR manual
//   model.save('model.h5')    → model.save('file://...')

const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });
const tf   = require('@tensorflow/tfjs');
const fs   = require('fs');
const Jimp = require('jimp');
const { buildLungMriModel } = require('./modelo');

// ── Configuración (equivale al bloque de parámetros del notebook) ─────────────
const IMG_SIZE     = parseInt(process.env.IMG_SIZE)   || 128;
const BATCH_SIZE   = parseInt(process.env.BATCH_SIZE) || 8;
const EPOCHS       = parseInt(process.env.EPOCHS)     || 1;
const DATA_FRACTION = parseFloat(process.env.DATA_FRACTION) || 0.25;
const MAX_IMAGES   = parseInt(process.env.MAX_IMAGES)   || 10;

// Rutas — equivalen a train_dir / val_dir del notebook
const TRAIN_DIR    = path.join(__dirname, '..', 'dataset', 'train');
const VAL_DIR      = path.join(__dirname, '..', 'dataset', 'validate');
const SAVE_DIR     = path.join(__dirname, '..', 'modelo_guardado');

// Clases en el mismo orden que Keras las detecta (orden alfabético)
// cancer=0, no_cancer=1  (Keras: alphabetical → cancer < no_cancer)
// PERO el notebook usa: class_mode='binary' donde cancer=1, no_cancer=0
// Mantenemos: { cancer: 1, no_cancer: 0 } para coincidir con el notebook
const CLASES = { no_cancer: 0, cancer: 1 };

console.log('══════════════════════════════════════════════');
console.log('   CNN — Cáncer Pulmonar MRI  |  Entrenamiento');
console.log('══════════════════════════════════════════════');
console.log(`📁 Train    : ${TRAIN_DIR}`);
console.log(`📁 Validate : ${VAL_DIR}`);
console.log(`🖼️  Tamaño   : ${IMG_SIZE}×${IMG_SIZE} px (grayscale)`);
console.log(`📦 Batch    : ${BATCH_SIZE} | 🔁 Épocas: ${EPOCHS}`);
console.log(`🧪 Dataset  : ${DATA_FRACTION * 100}% de las imágenes`);
console.log(`🔢 Máx imágenes por partición: ${MAX_IMAGES}`);
console.log('──────────────────────────────────────────────\n');

// ════════════════════════════════════════════════════════
//  SECCIÓN 1: CARGA Y PREPROCESAMIENTO DE IMÁGENES
//  Equivale a: ImageDataGenerator + flow_from_directory
//  color_mode='grayscale' → 1 canal
//  rescale=1./255         → normalización
// ════════════════════════════════════════════════════════

/**
 * Lee una imagen en escala de grises, la redimensiona a IMG_SIZE×IMG_SIZE
 * y la normaliza (divide entre 255). Equivale a:
 *   rescale=1./255, color_mode='grayscale'
 */
async function imagenAGrayscale(rutaArchivo, augmentar = false) {
  let img = await Jimp.read(rutaArchivo);
  img.resize(IMG_SIZE, IMG_SIZE);

  // Data augmentation — equivale a los parámetros del train_datagen:
  //   rotation_range=15, zoom_range=0.1,
  //   width_shift_range=0.1, height_shift_range=0.1,
  //   horizontal_flip=True
  if (augmentar) {
    // Volteo horizontal aleatorio (horizontal_flip=True)
    if (Math.random() > 0.5) img.mirror(true, false);

    // Rotación ±15° (rotation_range=15)
    const angulo = (Math.random() * 30) - 15;
    img.rotate(angulo, false);

    // Brillo aleatorio leve para simular zoom/shift
    const factor = 0.9 + Math.random() * 0.2;
    img.brightness(factor - 1);
  }

  // Convertir a escala de grises y extraer canal único
  img.grayscale();

  const buffer = new Float32Array(IMG_SIZE * IMG_SIZE * 1);
  let i = 0;
  img.scan(0, 0, img.bitmap.width, img.bitmap.height, (x, y, idx) => {
    buffer[i++] = img.bitmap.data[idx] / 255.0; // solo canal R (=G=B en gris)
  });

  return tf.tensor3d(buffer, [IMG_SIZE, IMG_SIZE, 1]);
}

/**
 * Carga todas las imágenes de una partición (train o validate).
 * Devuelve tensores X (imágenes) e Y (etiquetas binarias 0/1).
 *
 * Equivale a flow_from_directory con class_mode='binary'
 */
function mezclarArreglo(arr) {
  return arr.slice().sort(() => Math.random() - 0.5);
}

async function cargarParticion(dirBase, augmentar = false) {
  const tensoresX = [];
  const etiquetas = [];
  const conteo    = {};

  for (const [nombreClase, labelVal] of Object.entries(CLASES)) {
    const claseDir = path.join(dirBase, nombreClase);

    if (!fs.existsSync(claseDir)) {
      console.warn(`  ⚠️  Carpeta no encontrada: ${claseDir}`);
      continue;
    }

    let archivos = fs.readdirSync(claseDir)
      .filter(f => /\.(jpg|jpeg|png|bmp)$/i.test(f));

    if (DATA_FRACTION > 0 && DATA_FRACTION < 1) {
      const cantidad = Math.max(1, Math.floor(archivos.length * DATA_FRACTION));
      archivos = mezclarArreglo(archivos).slice(0, cantidad);
    }

    if (MAX_IMAGES > 0) {
      const maxPorClase = Math.max(1, Math.ceil(MAX_IMAGES / Object.keys(CLASES).length));
      archivos = mezclarArreglo(archivos).slice(0, maxPorClase);
    }

    conteo[nombreClase] = archivos.length;
    console.log(`  📂 ${nombreClase.padEnd(12)}: ${archivos.length} imágenes`);

    for (const archivo of archivos) {
      try {
        const tensor = await imagenAGrayscale(
          path.join(claseDir, archivo),
          augmentar
        );
        tensoresX.push(tensor);
        etiquetas.push([labelVal]); // etiqueta binaria: 0 o 1
      } catch (e) {
        console.warn(`    ⚠️  Saltando ${archivo}: ${e.message}`);
      }
    }
  }

  if (tensoresX.length === 0) {
    throw new Error(`No se encontraron imágenes en: ${dirBase}`);
  }

  const X = tf.stack(tensoresX);          // [N, 224, 224, 1]
  const Y = tf.tensor2d(etiquetas);       // [N, 1]
  tensoresX.forEach(t => t.dispose());

  return { X, Y, conteo };
}

// ════════════════════════════════════════════════════════
//  SECCIÓN 2: CLASS WEIGHTS
//  Equivale a: compute_class_weight(class_weight='balanced')
// ════════════════════════════════════════════════════════

/**
 * Calcula pesos de clase balanceados.
 * Fórmula: weight_i = total / (n_clases * count_i)
 *
 * Equivale a sklearn.utils.class_weight.compute_class_weight('balanced')
 */
function computarClassWeights(conteo) {
  const total   = Object.values(conteo).reduce((a, b) => a + b, 0);
  const nClases = Object.keys(conteo).length;
  const weights = {};

  for (const [clase, count] of Object.entries(conteo)) {
    const labelIdx   = CLASES[clase];
    weights[labelIdx] = total / (nClases * count);
  }

  return weights;
}

// ════════════════════════════════════════════════════════
//  SECCIÓN 3: DISTRIBUCIÓN DEL DATASET
//  Equivale al bloque "Dataset Distribution" del notebook
//  (aquí se imprime en consola en lugar de gráfico)
// ════════════════════════════════════════════════════════

function mostrarDistribucion(conteoTrain, conteoVal) {
  console.log('\n📊 Distribución del Dataset:');
  console.log('┌─────────────┬──────────────┬────────────────┐');
  console.log('│ Clase       │ Train        │ Validate       │');
  console.log('├─────────────┼──────────────┼────────────────┤');
  for (const clase of Object.keys(CLASES)) {
    const tr = String(conteoTrain[clase] ?? 0).padStart(6);
    const vl = String(conteoVal[clase]   ?? 0).padStart(6);
    console.log(`│ ${clase.padEnd(11)} │     ${tr}       │       ${vl}       │`);
  }
  console.log('└─────────────┴──────────────┴────────────────┘\n');
}

// ════════════════════════════════════════════════════════
//  SECCIÓN 4: ENTRENAMIENTO CON CALLBACKS
//  Equivale a:
//    EarlyStopping(monitor='val_loss', patience=10)
//    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
//    model.fit(..., epochs=50, class_weight=class_weight_dict)
// ════════════════════════════════════════════════════════

async function entrenar() {

  // 1. Cargar datos
  console.log('📊 Cargando datos de entrenamiento...');
  const { X: X_train, Y: Y_train, conteo: conteoTrain } =
    await cargarParticion(TRAIN_DIR, true);   // augmentar=true

  console.log('\n📊 Cargando datos de validación...');
  const { X: X_val, Y: Y_val, conteo: conteoVal } =
    await cargarParticion(VAL_DIR, false);    // sin augmentation

  // 2. Mostrar distribución (equivale al bloque "Dataset Distribution")
  mostrarDistribucion(conteoTrain, conteoVal);

  // 3. Class weights (equivale a compute_class_weight)
  const classWeights = computarClassWeights(conteoTrain);
  console.log('⚖️  Class Weights:', classWeights);
  console.log('   (class_weight="balanced" — compensa el desbalance)\n');

  // 4. Construir modelo
  const model = buildLungMriModel(IMG_SIZE);

  // ReduceLROnPlateau: monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
  let lrActual       = 1e-4;
  let pacienciaLR    = 0;
  const PATIENCE_LR  = 5;
  const FACTOR_LR    = 0.5;
  const MIN_LR       = 1e-6;
  const COOLDOWN_LR  = 2;
  let cooldownRestante = 0;

  // TF.js no soporta sampleWeight directamente en esta versión,
  // así que usamos una pérdida binaria ponderada manualmente.
  const weightedBinaryCrossentropy = (yTrue, yPred) => {
    return tf.tidy(() => {
      const w0 = tf.scalar(classWeights[0]);
      const w1 = tf.scalar(classWeights[1]);
      const weights = yTrue.mul(w1.sub(w0)).add(w0);
      const epsilon = 1e-7;
      const clippedPred = yPred.clipByValue(epsilon, 1 - epsilon);
      const loss = yTrue.mul(clippedPred.log())
        .add(tf.scalar(1).sub(yTrue).mul(tf.scalar(1).sub(clippedPred).log()))
        .mul(tf.scalar(-1));
      return loss.mul(weights).mean();
    });
  };

  model.compile({
    optimizer: tf.train.adam(lrActual),
    loss: weightedBinaryCrossentropy,
    metrics: ['accuracy']
  });

  model.summary();
  console.log('');

  // 5. Preparar directorio de guardado
  if (!fs.existsSync(SAVE_DIR)) {
    fs.mkdirSync(SAVE_DIR, { recursive: true });
  }

  // Helper: guardar modelo a disco sin depender de tfjs-node
  async function guardarModeloEnCarpeta(model, dir) {
    // Use withSaveHandler to get modelArtifacts
    const saveResult = await model.save(tf.io.withSaveHandler(async (modelArtifacts) => {
      // modelArtifacts: {modelTopology, weightSpecs, weightData}
      const modelJson = {
        format: 'layers-model',
        generatedBy: 'custom-save-handler',
        modelTopology: modelArtifacts.modelTopology,
        weightsManifest: [
          {
            paths: ['weights.bin'],
            weights: modelArtifacts.weightSpecs
          }
        ]
      };

      // Escribir model.json
      fs.writeFileSync(path.join(dir, 'model.json'), JSON.stringify(modelJson, null, 2), 'utf8');

      // Escribir weights.bin
      const buf = Buffer.from(modelArtifacts.weightData);
      fs.writeFileSync(path.join(dir, 'weights.bin'), buf);

      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON',
          modelTopologyBytes: modelArtifacts.modelTopology ? JSON.stringify(modelArtifacts.modelTopology).length : 0,
          weightDataBytes: buf.length
        }
      };
    }));
    return saveResult;
  }

  // ── Variables para callbacks manuales ──────────────────────────────────────
  // EarlyStopping: monitor='val_loss', patience=10, min_delta=0.001
  let mejorValLoss   = Infinity;
  let pacienciaES    = 0;
  const PATIENCE_ES  = 10;
  const MIN_DELTA    = 0.001;

  // Historial (para guardar métricas y mostrarlas al frontend)
  const historial = { accuracy: [], val_accuracy: [], loss: [], val_loss: [] };

  console.log('🚀 Iniciando entrenamiento...\n');

  for (let epoca = 1; epoca <= EPOCHS; epoca++) {
    console.log(`\n▶️  Iniciando época ${epoca}/${EPOCHS}...`);
    const totalBatches = Math.ceil(X_train.shape[0] / BATCH_SIZE);
    console.log(`   📌 Lotes estimados: ${totalBatches}`);

    // ── Entrenamiento de una época ────────────────────────────────────────────
    console.log('   ⏳ Ejecutando model.fit...');
    const inicioEpoca = Date.now();
    const history = await model.fit(X_train, Y_train, {
      epochs:         1,
      batchSize:      BATCH_SIZE,
      validationData: [X_val, Y_val],
      shuffle:        true,
      verbose:        0,
      callbacks: {
        onTrainBegin: async () => {
          console.log('   ▶️  Training iniciado');
        },
        onBatchBegin: async (batch) => {
          process.stdout.write(`   🔄 Iniciando lote ${batch + 1}/${totalBatches}...   \r`);
        },
        onBatchEnd: async (batch, logs) => {
          process.stdout.write(
            `   ⏳ lote ${batch + 1}/${totalBatches}  loss=${logs.loss.toFixed(4)}  ` +
            `acc=${(logs.acc ?? logs.accuracy ?? 0).toFixed(4)}   \r`
          );
        },
        onEpochEnd: async (_epoch, logs) => {
          const duracion = ((Date.now() - inicioEpoca) / 1000).toFixed(1);
          console.log(`\n   ✅ model.fit completado en ${duracion}s | loss=${logs.loss.toFixed(4)} | val_loss=${logs.val_loss.toFixed(4)}`);
        }
      }
    });

    console.log(`   ✅ Época ${epoca} completada | loss=${history.history.loss[0].toFixed(4)} | val_loss=${history.history.val_loss[0].toFixed(4)}`);

    // Extraer métricas
    const trainAcc = history.history.accuracy?.[0] ?? history.history.acc?.[0] ?? 0;
    const valAcc   = history.history.val_accuracy?.[0] ?? history.history.val_acc?.[0] ?? 0;
    const loss     = history.history.loss[0];
    const valLoss  = history.history.val_loss[0];

    historial.accuracy.push(trainAcc);
    historial.val_accuracy.push(valAcc);
    historial.loss.push(loss);
    historial.val_loss.push(valLoss);

    // ── ReduceLROnPlateau ─────────────────────────────────────────────────────
    let lrMsg = '';
    if (cooldownRestante > 0) {
      cooldownRestante--;
    } else if (valLoss < mejorValLoss - MIN_DELTA) {
      pacienciaLR = 0;
    } else {
      pacienciaLR++;
      if (pacienciaLR >= PATIENCE_LR) {
        const nuevoLR = Math.max(lrActual * FACTOR_LR, MIN_LR);
        if (nuevoLR < lrActual) {
          lrActual         = nuevoLR;
          cooldownRestante = COOLDOWN_LR;
          // Actualizar optimizer
          model.optimizer.learningRate = lrActual;
          lrMsg = ` ⬇️  LR→${lrActual.toExponential(1)}`;
        }
        pacienciaLR = 0;
      }
    }

    // ── EarlyStopping ─────────────────────────────────────────────────────────
    const indicador = valLoss < mejorValLoss - MIN_DELTA ? '🟢' : '🔴';
    let esMsg = '';

    if (valLoss < mejorValLoss - MIN_DELTA) {
      mejorValLoss = valLoss;
      pacienciaES  = 0;
      await guardarModeloEnCarpeta(model, SAVE_DIR);
      esMsg = ' 💾 guardado';
    } else {
      pacienciaES++;
      esMsg = ` (paciencia ${pacienciaES}/${PATIENCE_ES})`;
    }

    console.log(
      `Época ${String(epoca).padStart(2,'0')}/${EPOCHS}  ${indicador}  ` +
      `loss: ${loss.toFixed(4)}  acc: ${(trainAcc*100).toFixed(1)}%  |  ` +
      `val_loss: ${valLoss.toFixed(4)}  val_acc: ${(valAcc*100).toFixed(1)}%` +
      lrMsg + esMsg
    );

    if (pacienciaES >= PATIENCE_ES) {
      console.log(`\n⏹️  Early stopping — sin mejora en ${PATIENCE_ES} épocas consecutivas`);
      console.log('   (restore_best_weights=True → modelo ya guardado en mejor época)');
      break;
    }
  }

  // ── Guardar metadatos ─────────────────────────────────────────────────────
  fs.writeFileSync(
    path.join(SAVE_DIR, 'meta.json'),
    JSON.stringify({
      clases:    CLASES,         // { no_cancer: 0, cancer: 1 }
      imgSize:   IMG_SIZE,
      threshold: 0.5,            // umbral sigmoid: >=0.5 → cancer
      historial,
      conteoTrain,
      conteoVal
    }, null, 2)
  );

  // ── Evaluación final ──────────────────────────────────────────────────────
  console.log('\n📊 Evaluación final en validación:');
  const [evalLoss, evalAcc] = model.evaluate(X_val, Y_val, { verbose: 0 });
  console.log(`  Loss     : ${(await evalLoss.data())[0].toFixed(4)}`);
  console.log(`  Accuracy : ${((await evalAcc.data())[0] * 100).toFixed(2)}%`);

  X_train.dispose(); Y_train.dispose();
  X_val.dispose();   Y_val.dispose();

  console.log('\n══════════════════════════════════════════════');
  console.log('  ✅ Entrenamiento completado');
  console.log(`  📁 Modelo guardado en: ${SAVE_DIR}`);
  console.log('══════════════════════════════════════════════\n');
}

entrenar().catch(err => {
  console.error('\n❌ Error:', err.message);
  process.exit(1);
});