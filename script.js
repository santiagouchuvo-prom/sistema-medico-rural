// =======================================
// 1. BASE DE DATOS INDEXEDDB (DEXIE.JS)
// =======================================
const db = new Dexie("SistemaMedicoRuralDB");

// El esquema NO cambia: los datos que ya tengas guardados siguen funcionando.
db.version(1).stores({
    pacientes: '++id, nombre, numero, rh, ciudad',
    citas: '++id, pacienteId, prioridad, estado, fecha'
});

// Sembrado de paciente predeterminado
db.on("ready", async () => {
    const count = await db.pacientes.count();
    if (count === 0) {
        await db.pacientes.add({
            nombre: 'Santiago',
            documento: 'C.C',
            numero: '1032941426',
            edad: 18,
            fecha: '2008-03-31',
            rh: 'O+',
            direccion: 'Vereda Yerbabuena',
            ciudad: 'Chía',
            contacto: 'Mamá - 3000001212',
            telefono: '3105838753',
            alergias: 'Penicilina / AINEs',
            enfermedades: 'Fisura de tibia (Tratada)',
            medicamentos: 'Ninguno actual',
            historial: 'Ruptura de peroné y fisura de tibia. Paciente registrado en puesto rural.',
            foto: 'https://i.imgur.com/9Xn4F6L.png',
            descriptor: null
        });
    }
});

// Escapa texto antes de insertarlo con innerHTML (evita inyección de HTML/JS)
function esc(valor) {
    return String(valor ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// =======================================
// 2. MODELOS DE IA FACIAL
// =======================================
// Primero intenta la carpeta local ./models (funciona sin internet) y luego el CDN.
// Estos modelos corresponden a la librería @vladmandic/face-api que ya cargas en el HTML.
const MODEL_URLS = [
    './models',
    'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model'
];
const UMBRAL_COINCIDENCIA = 0.55; // distancia máxima para considerar "mismo rostro" (0.6 es el valor por defecto de la librería)
const MAX_LADO_IMAGEN = 800;      // las fotos se reducen a este tamaño antes de analizarlas y guardarlas

let iaLista = false;
let iaPromesa = null;

function setEstadoRegistro(texto, clase = 'text-muted') {
    const el = document.getElementById('estadoModeloRegistro');
    if (!el) return;
    el.textContent = texto;
    el.className = `small fw-bold ${clase}`;
}

function setEstadoSOS(texto, clase = 'text-muted') {
    const el = document.getElementById('sosEstadoTexto');
    if (!el) return;
    el.textContent = texto;
    el.className = clase;
}

async function prepararBackendIA() {
    if (typeof faceapi === 'undefined' || !faceapi.tf) return;
    try {
        let ok = false;
        try { ok = await faceapi.tf.setBackend('webgl'); } catch (e) { ok = false; }
        if (!ok) await faceapi.tf.setBackend('cpu'); // equipos sin aceleración gráfica
        await faceapi.tf.ready();
    } catch (e) {
        console.warn('No se pudo configurar el backend de IA, se usará el predeterminado:', e);
    }
}

// Carga los modelos UNA sola vez. Si ya están cargados o cargando, reutiliza la misma promesa.
// Si falla, permite reintentar.
function inicializarModelosIA() {
    if (iaLista) return Promise.resolve(true);
    if (iaPromesa) return iaPromesa;

    iaPromesa = (async () => {
        setEstadoRegistro('⏳ Cargando inteligencia artificial para huella biométrica...');

        if (typeof faceapi === 'undefined') {
            console.error('La librería face-api no se cargó (¿sin internet?).');
            setEstadoRegistro('❌ No se cargó la librería de IA. Verifique la conexión o use copias locales.', 'text-danger');
            return false;
        }

        await prepararBackendIA();

        for (const url of MODEL_URLS) {
            try {
                await Promise.all([
                    faceapi.nets.ssdMobilenetv1.loadFromUri(url),
                    faceapi.nets.faceLandmark68Net.loadFromUri(url),   // nombre correcto (antes: faceLandmarkNet, que no existe)
                    faceapi.nets.faceRecognitionNet.loadFromUri(url)
                ]);
                iaLista = true;
                console.log('Modelos de IA facial cargados desde:', url);
                setEstadoRegistro('✅ IA lista para generar huella biométrica.', 'text-success');
                return true;
            } catch (e) {
                console.warn(`No se pudieron cargar los modelos desde ${url}:`, e);
            }
        }

        setEstadoRegistro('❌ No se pudieron cargar los modelos de IA. Revise la carpeta /models o la conexión a internet.', 'text-danger');
        return false;
    })();

    iaPromesa.then(ok => { if (!ok) iaPromesa = null; }); // permite reintentar tras un fallo
    return iaPromesa;
}

// ---------- Utilidades de imagen y detección ----------
function archivoADataURL(archivo) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(archivo);
    });
}

// Carga una imagen (dataURL o URL) y la dibuja en un canvas reducido a MAX_LADO_IMAGEN
async function imagenACanvas(src, maxLado = MAX_LADO_IMAGEN) {
    const img = new Image();
    img.src = src;
    await img.decode();

    const escala = Math.min(1, maxLado / Math.max(img.naturalWidth, img.naturalHeight));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(img.naturalWidth * escala);
    canvas.height = Math.round(img.naturalHeight * escala);
    canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas;
}

async function detectarRostros(canvas) {
    return faceapi
        .detectAllFaces(canvas, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
        .withFaceLandmarks()
        .withFaceDescriptors();
}

// Si hay varios rostros, toma el más grande (el más cercano a la cámara)
function rostroPrincipal(detecciones) {
    return detecciones.reduce((mayor, d) =>
        d.detection.box.area > mayor.detection.box.area ? d : mayor
    );
}

// Procesa la foto de un paciente: reduce, valida que haya UN rostro y genera la huella.
// Devuelve { foto, descriptor } o lanza un Error con un mensaje listo para mostrar.
async function procesarFotoPaciente(archivo) {
    let canvas;
    try {
        canvas = await imagenACanvas(await archivoADataURL(archivo));
    } catch (err) {
        console.error('Error leyendo la imagen:', err);
        throw new Error('No se pudo leer la imagen. Use un archivo JPG o PNG válido.');
    }

    let detecciones;
    try {
        detecciones = await detectarRostros(canvas);
    } catch (err) {
        console.error('Error analizando biometría:', err);
        throw new Error('Error al procesar la fotografía con la IA.');
    }

    if (detecciones.length === 0) {
        throw new Error('No se detectó un rostro claro en la foto. Use una imagen con buena iluminación y el rostro de frente.');
    }
    if (detecciones.length > 1) {
        throw new Error('Se detectaron varios rostros en la foto. Use una fotografía donde aparezca solo el paciente.');
    }

    return {
        foto: canvas.toDataURL('image/jpeg', 0.85),
        descriptor: Array.from(detecciones[0].descriptor)
    };
}

// =======================================
// 3. AUTENTICACIÓN Y ROLES
// =======================================
const usuarios = {
    admin: { usuario: 'admin', password: '123', rol: 'admin' },
    trabajador: { usuario: 'medico', password: '123', rol: 'trabajador' }
};

function login() {
    const u = document.getElementById('usuario').value;
    const p = document.getElementById('password').value;

    if (u === usuarios.admin.usuario && p === usuarios.admin.password) {
        localStorage.setItem('rol', 'admin');
        window.location.href = 'dashboard.html';
    } else if (u === usuarios.trabajador.usuario && p === usuarios.trabajador.password) {
        localStorage.setItem('rol', 'trabajador');
        window.location.href = 'dashboard.html';
    } else {
        alert('Usuario o contraseña incorrectos');
    }
}

function cerrarSesion() {
    localStorage.removeItem('rol');
    window.location.href = 'index.html';
}

function verificarRolUI() {
    const rol = localStorage.getItem('rol');
    const badge = document.getElementById('badgeRol');
    const btnReg = document.getElementById('btnRegistro');

    if (badge) badge.innerText = `Rol: ${rol ? rol.toUpperCase() : 'Invitado'}`;
    if (btnReg && rol === 'trabajador') {
        btnReg.style.display = 'none';
    }
}

// =======================================
// 4. GESTIÓN DE PACIENTES (INDEXEDDB)
// =======================================
async function cargarPacientesDashboard() {
    const container = document.getElementById('patientsContainer');
    const selectCita = document.getElementById('citaPaciente');
    const rol = localStorage.getItem('rol');
    if (!container) return;

    const pacientes = await db.pacientes.toArray();
    let html = '';
    let opciones = '<option value="">Seleccione Paciente...</option>';

    pacientes.forEach(p => {
        const botonEditarHTML = rol === 'admin'
            ? `<button class="btn btn-sm btn-warning fw-bold mt-2 w-100" onclick="abrirModalEditar(${p.id})">✏️ Editar Paciente</button>`
            : '';

        const avisoHuella = Array.isArray(p.descriptor)
            ? ''
            : '<span class="badge bg-secondary mb-2">Sin huella facial (el SOS no lo reconocerá)</span>';

        html += `
            <div class="col-md-6 mb-3">
                <div class="card shadow-sm h-100">
                    <div class="card-body text-center">
                        <img src="${esc(p.foto)}" class="rounded-circle mb-2" width="90" height="90" style="object-fit:cover;">
                        <h5 class="card-title fw-bold text-primary mb-1">${esc(p.nombre)}</h5>
                        <p class="mb-1 small"><strong>RH:</strong> <span class="badge bg-danger">${esc(p.rh)}</span> | <strong>Edad:</strong> ${esc(p.edad)}</p>
                        <p class="mb-2 small text-muted"><strong>Ubicación:</strong> ${esc(p.ciudad)}</p>
                        ${avisoHuella}
                        <a href="patient.html?id=${p.id}" class="btn btn-sm btn-outline-primary fw-bold w-100 mb-1">Ver Expediente</a>
                        ${botonEditarHTML}
                    </div>
                </div>
            </div>
        `;

        opciones += `<option value="${p.id}">${esc(p.nombre)} (${esc(p.numero)})</option>`;
    });

    container.innerHTML = html || '<p class="text-muted mb-0">No hay pacientes registrados.</p>';
    if (selectCita) selectCita.innerHTML = opciones;
}

// REGISTRO EXIGIENDO HUELLA BIOMÉTRICA VÁLIDA
async function guardarNuevoPaciente(e) {
    e.preventDefault();

    const btn = e.submitter || e.target.querySelector('button[type="submit"]');
    const textoOriginal = btn ? btn.textContent : '';
    const setBtn = (texto, deshabilitado) => {
        if (!btn) return;
        btn.textContent = texto;
        btn.disabled = deshabilitado;
    };

    const archivo = document.getElementById('foto').files[0];
    if (!archivo) {
        alert('Por favor adjunte una fotografía.');
        return;
    }

    const numero = document.getElementById('numero').value.trim();

    try {
        // Si los modelos aún cargan, espera (no rechaza); si fallaron, reintenta
        setBtn('⏳ Cargando IA...', true);
        if (!(await inicializarModelosIA())) {
            alert('No se pudieron cargar los modelos de IA. Revise la carpeta /models o la conexión a internet e intente de nuevo.');
            return;
        }

        const repetido = await db.pacientes.where('numero').equals(numero).first();
        if (repetido) {
            alert(`Ya existe un paciente con el documento ${numero}: ${repetido.nombre}.`);
            return;
        }

        setBtn('⏳ Analizando rostro...', true);
        const { foto, descriptor } = await procesarFotoPaciente(archivo);

        await db.pacientes.add({
            nombre: document.getElementById('nombre').value.trim(),
            documento: document.getElementById('documento').value,
            numero: numero,
            edad: Number(document.getElementById('edad').value),
            fecha: document.getElementById('fecha').value,
            rh: document.getElementById('rh').value,
            telefono: document.getElementById('telefono').value.trim(),
            ciudad: document.getElementById('ciudad').value.trim(),
            direccion: document.getElementById('direccion').value.trim(),
            alergias: document.getElementById('alergias').value.trim(),
            enfermedades: document.getElementById('enfermedades').value.trim(),
            medicamentos: document.getElementById('medicamentos').value.trim(),
            contacto: document.getElementById('contacto').value.trim(),
            historial: document.getElementById('historial').value.trim(),
            foto: foto,
            descriptor: descriptor
        });

        alert('¡Paciente registrado con éxito y huella biométrica guardada!');
        window.location.href = 'dashboard.html';
    } catch (err) {
        console.error(err);
        alert(err.message || 'Error al guardar el paciente.');
    } finally {
        setBtn(textoOriginal, false);
    }
}

// =======================================
// 5. EDICIÓN DE PACIENTES Y FOTOGRAFÍA
// =======================================
async function abrirModalEditar(id) {
    const paciente = await db.pacientes.get(id);
    if (!paciente) return;

    document.getElementById('editId').value = paciente.id;
    document.getElementById('editNombre').value = paciente.nombre;
    document.getElementById('editTelefono').value = paciente.telefono;
    document.getElementById('editAlergias').value = paciente.alergias;
    document.getElementById('editEnfermedades').value = paciente.enfermedades;
    document.getElementById('editMedicamentos').value = paciente.medicamentos;
    document.getElementById('editContacto').value = paciente.contacto;
    document.getElementById('editHistorial').value = paciente.historial;
    document.getElementById('editFoto').value = '';

    bootstrap.Modal.getOrCreateInstance(document.getElementById('editarModal')).show();
}

async function guardarEdicionPaciente(e) {
    e.preventDefault();

    const id = Number(document.getElementById('editId').value);
    const archivo = document.getElementById('editFoto').files[0];

    const actual = await db.pacientes.get(id);
    if (!actual) return;

    let foto = actual.foto;
    let descriptor = actual.descriptor;

    // Si cambian la foto, la huella se regenera SIEMPRE (así foto y huella nunca quedan desfasadas).
    if (archivo) {
        try {
            if (!(await inicializarModelosIA())) {
                alert('No se pudieron cargar los modelos de IA. No se puede cambiar la fotografía en este momento.');
                return;
            }
            const resultado = await procesarFotoPaciente(archivo);
            foto = resultado.foto;
            descriptor = resultado.descriptor;
        } catch (err) {
            alert(err.message || 'Error al procesar la nueva fotografía.');
            return;
        }
    }

    await db.pacientes.update(id, {
        nombre: document.getElementById('editNombre').value.trim(),
        telefono: document.getElementById('editTelefono').value.trim(),
        alergias: document.getElementById('editAlergias').value.trim(),
        enfermedades: document.getElementById('editEnfermedades').value.trim(),
        medicamentos: document.getElementById('editMedicamentos').value.trim(),
        contacto: document.getElementById('editContacto').value.trim(),
        historial: document.getElementById('editHistorial').value.trim(),
        foto: foto,
        descriptor: descriptor
    });

    alert('¡Perfil del paciente actualizado con éxito!');
    bootstrap.Modal.getOrCreateInstance(document.getElementById('editarModal')).hide();
    cargarPacientesDashboard();
}

// =======================================
// 6. MÓDULO DE CITAS CON TRIAJE
// =======================================
async function agendarCita(e) {
    e.preventDefault();
    const pacienteId = Number(document.getElementById('citaPaciente').value);
    const prioridad = document.getElementById('citaPrioridad').value;
    const motivo = document.getElementById('citaMotivo').value;

    const paciente = await db.pacientes.get(pacienteId);
    if (!paciente) return;

    let centroSalud = "Puesto de Salud Rural San José";
    const ciudad = (paciente.ciudad || '').toLowerCase();
    if (ciudad.includes("cajicá") || ciudad.includes("cajica")) {
        centroSalud = "Puesto de Salud Capellanía - Cajicá";
    } else if (ciudad.includes("chía") || ciudad.includes("chia")) {
        centroSalud = "Centro de Salud Veredal Yerbabuena - Chía";
    }

    const fechaHora = new Date();
    if (prioridad === "Alta") fechaHora.setHours(fechaHora.getHours() + 1);
    else if (prioridad === "Media") fechaHora.setDate(fechaHora.getDate() + 1);
    else fechaHora.setDate(fechaHora.getDate() + 3);

    const nuevaCita = {
        pacienteId: paciente.id,
        pacienteNombre: paciente.nombre,
        prioridad: prioridad,
        motivo: motivo,
        centroSalud: centroSalud,
        fechaHora: fechaHora.toLocaleString(),
        estado: 'En espera'
    };

    await db.citas.add(nuevaCita);
    alert(`Cita programada con éxito.\nCentro asignado: ${centroSalud}\nAtención estimada: ${nuevaCita.fechaHora}`);
    document.getElementById('formCita').reset();
    cargarCitas();
}

async function cargarCitas() {
    const container = document.getElementById('citasContainer');
    if (!container) return;

    const citas = await db.citas.toArray();
    if (citas.length === 0) {
        container.innerHTML = '<li class="list-group-item text-muted small">No hay citas agendadas.</li>';
        return;
    }

    let html = '';
    citas.forEach(c => {
        const badgeColor = c.prioridad === 'Alta' ? 'bg-danger' : (c.prioridad === 'Media' ? 'bg-warning text-dark' : 'bg-success');
        html += `
            <li class="list-group-item">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <strong class="small">${esc(c.pacienteNombre)}</strong>
                    <span class="badge ${badgeColor}">${esc(c.prioridad)}</span>
                </div>
                <p class="mb-1 text-muted" style="font-size:0.8rem;">📍 ${esc(c.centroSalud)}</p>
                <div class="d-flex justify-content-between align-items-center" style="font-size:0.75rem;">
                    <span>🕒 ${esc(c.fechaHora)}</span>
                    <span class="text-primary fw-bold">${esc(c.estado)}</span>
                </div>
            </li>
        `;
    });
    container.innerHTML = html;
}

// =======================================
// 7. MÓDULO DE RECONOCIMIENTO SOS
// =======================================
let sosStream = null;
let sosBuscando = false;

async function iniciarCamaraSOS() {
    try {
        sosStream = await navigator.mediaDevices.getUserMedia({ video: true });
        document.getElementById('sosVideo').srcObject = sosStream;
        return true;
    } catch (err) {
        console.log('Cámara deshabilitada:', err);
        return false;
    }
}

async function abrirModalSOS() {
    bootstrap.Modal.getOrCreateInstance(document.getElementById('sosModal')).show();

    detenerCamaraSOS(); // por si quedó una cámara abierta de antes
    const inputArchivo = document.getElementById('fotoArchivoSOS');
    if (inputArchivo) inputArchivo.value = '';

    setEstadoSOS('Iniciando cámara y verificando IA...');

    // Cámara y modelos arrancan en paralelo para ganar tiempo
    const camaraPromesa = iniciarCamaraSOS();
    const iaOk = await inicializarModelosIA();
    const camaraOk = await camaraPromesa;

    if (!iaOk) {
        setEstadoSOS('❌ No se pudieron cargar los modelos de IA. Revise la carpeta /models o la conexión.', 'text-danger');
        return;
    }
    if (camaraOk) {
        setEstadoSOS('Cámara activa. Apunte al rostro del paciente.');
    } else {
        setEstadoSOS('Cámara no disponible. Use la opción de subir fotografía.', 'text-warning');
    }
}

function detenerCamaraSOS() {
    if (sosStream) {
        sosStream.getTracks().forEach(t => t.stop());
        sosStream = null;
    }
    const video = document.getElementById('sosVideo');
    if (video) video.srcObject = null;
}

async function capturarYSOS() {
    const video = document.getElementById('sosVideo');
    if (!sosStream || !video.videoWidth) {
        setEstadoSOS('La cámara aún no está lista. Espere un momento o suba una fotografía.', 'text-warning');
        return;
    }

    const canvas = document.getElementById('sosCanvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    await ejecutarBusquedaInteligenteSOS(canvas.toDataURL('image/jpeg', 0.9));
}

async function cargarArchivoSOS(e) {
    const file = e.target.files[0];
    if (!file) return;

    try {
        await ejecutarBusquedaInteligenteSOS(await archivoADataURL(file));
    } catch (err) {
        console.error(err);
        setEstadoSOS('No se pudo leer la imagen.', 'text-danger');
    } finally {
        e.target.value = '';
    }
}

// Búsqueda estricta por distancia entre huellas faciales.
// El modal permanece abierto si no hay coincidencia, para poder reintentar de inmediato.
async function ejecutarBusquedaInteligenteSOS(imagenSrc) {
    if (sosBuscando) return;
    sosBuscando = true;

    try {
        if (!(await inicializarModelosIA())) {
            setEstadoSOS('❌ Los modelos de IA no están disponibles.', 'text-danger');
            return;
        }

        const todos = await db.pacientes.toArray();
        if (todos.length === 0) {
            setEstadoSOS('No hay pacientes registrados en el sistema local.', 'text-warning');
            return;
        }
        const pacientes = todos.filter(p => Array.isArray(p.descriptor));
        if (pacientes.length === 0) {
            setEstadoSOS('Ningún paciente tiene huella facial. Registre o edite pacientes con fotografía.', 'text-warning');
            return;
        }

        setEstadoSOS('⏳ Analizando rostro...');
        const canvas = await imagenACanvas(imagenSrc);
        const detecciones = await detectarRostros(canvas);

        if (detecciones.length === 0) {
            setEstadoSOS('⚠️ No se detectó un rostro claro. Mejore la iluminación y apunte de frente.', 'text-warning');
            return;
        }
        const rostro = rostroPrincipal(detecciones);

        let mejorMatch = null;
        let menorDistancia = UMBRAL_COINCIDENCIA;

        pacientes.forEach(p => {
            const distancia = faceapi.euclideanDistance(rostro.descriptor, new Float32Array(p.descriptor));
            console.log(`Distancia con ${p.nombre}:`, distancia.toFixed(3));
            if (distancia < menorDistancia) {
                menorDistancia = distancia;
                mejorMatch = p;
            }
        });

        if (mejorMatch) {
            detenerCamaraSOS();
            setEstadoSOS(`✅ Coincidencia: ${mejorMatch.nombre}. Abriendo expediente...`, 'text-success');
            window.location.href = `patient.html?id=${mejorMatch.id}&modo=emergencia`;
        } else {
            setEstadoSOS('❌ El rostro NO coincide con ningún paciente registrado. Intente de nuevo con mejor luz.', 'text-danger');
        }
    } catch (error) {
        console.error('Error en análisis IA SOS:', error);
        setEstadoSOS('Ocurrió un error al procesar el reconocimiento facial.', 'text-danger');
    } finally {
        sosBuscando = false;
    }
}

// =======================================
// 8. CARGA DE EXPEDIENTE / VISTA SOS
// =======================================
async function cargarExpedientePaciente() {
    const info = document.getElementById('patientInfo');
    if (!info) return;

    const params = new URLSearchParams(window.location.search);
    const id = Number(params.get('id'));
    const esEmergencia = params.get('modo') === 'emergencia';

    const p = await db.pacientes.get(id);

    if (!p) {
        info.innerHTML = `<div class="alert alert-danger">Paciente no encontrado en el sistema local.</div>`;
        return;
    }

    if (esEmergencia) {
        info.innerHTML = `
            <div class="alert alert-danger text-center py-3 mb-4 shadow-sm" style="border-radius:12px;">
                <h2 class="fw-bold mb-0">🚨 ATENCIÓN MÉDICA DE EMERGENCIA 🚨</h2>
                <p class="mb-0">Identificación Biométrica por IA Verificada - Paciente Crítico</p>
            </div>

            <div class="row align-items-center">
                <div class="col-md-4 text-center border-end">
                    <img src="${esc(p.foto)}" width="200" height="200" class="rounded-circle shadow border border-danger border-4 mb-3" style="object-fit:cover;">
                    <h3 class="text-danger fw-bold">${esc(p.nombre)}</h3>
                    <p class="text-muted mb-0">Doc: ${esc(p.documento)} ${esc(p.numero)}</p>
                    <p class="text-muted">Edad: ${esc(p.edad)} años</p>
                </div>

                <div class="col-md-8">
                    <div class="card bg-danger-subtle border-danger p-3 mb-3 shadow-sm">
                        <h4 class="text-danger fw-bold">⚕️ DATOS CRÍTICOS DE SUPERVIVENCIA</h4>
                        <hr class="my-2">
                        <p class="fs-4 mb-2"><strong>Grupo Sanguíneo:</strong> <span class="badge bg-danger fs-3">${esc(p.rh)}</span></p>
                        <p class="fs-5 mb-2"><strong>Alergias Severas:</strong> <span class="text-danger fw-bold">${esc(p.alergias)}</span></p>
                        <p class="fs-5 mb-2"><strong>Enfermedades / Antecedentes Graves:</strong> <span class="fw-bold">${esc(p.enfermedades)}</span></p>
                        <p class="fs-5 mb-0"><strong>Medicamentos Actuales:</strong> ${esc(p.medicamentos)}</p>
                    </div>

                    <div class="card bg-warning-subtle border-warning p-3 shadow-sm">
                        <h5 class="fw-bold text-dark">📞 Contacto Directo de Emergencia</h5>
                        <p class="fs-5 mb-1"><strong>Pariente / Referencia:</strong> ${esc(p.contacto)}</p>
                        <a href="tel:${esc(p.telefono)}" class="btn btn-dark fw-bold w-100 py-2">📞 Llamar al Contacto (${esc(p.telefono)})</a>
                    </div>
                </div>
            </div>

            <div class="text-center mt-4">
                <a href="dashboard.html" class="btn btn-secondary btn-lg fw-bold">Regresar al Dashboard</a>
            </div>
        `;
    } else {
        info.innerHTML = `
            <div class="row">
                <div class="col-md-4 text-center">
                    <img src="${esc(p.foto)}" width="220" height="220" class="rounded shadow mb-3" style="object-fit:cover;">
                    <h2 class="text-primary fw-bold">${esc(p.nombre)}</h2>
                    <span class="badge bg-danger fs-5 mb-3">RH: ${esc(p.rh)}</span>
                </div>
                <div class="col-md-8">
                    <h4 class="fw-bold border-bottom pb-2">Información del Paciente</h4>
                    <ul class="list-group list-group-flush mb-4">
                        <li class="list-group-item"><strong>Documento:</strong> ${esc(p.documento)} - ${esc(p.numero)}</li>
                        <li class="list-group-item"><strong>Edad:</strong> ${esc(p.edad)} años (${esc(p.fecha)})</li>
                        <li class="list-group-item"><strong>Ubicación:</strong> ${esc(p.direccion)}, ${esc(p.ciudad)}</li>
                        <li class="list-group-item"><strong>Teléfono:</strong> ${esc(p.telefono)}</li>
                        <li class="list-group-item text-danger"><strong>Alergias:</strong> ${esc(p.alergias)}</li>
                        <li class="list-group-item text-danger"><strong>Enfermedades / Antecedentes:</strong> ${esc(p.enfermedades)}</li>
                        <li class="list-group-item"><strong>Medicamentos:</strong> ${esc(p.medicamentos)}</li>
                    </ul>
                    <h4 class="fw-bold border-bottom pb-2">Historia Clínica</h4>
                    <div class="alert alert-info shadow-sm">${esc(p.historial)}</div>
                    <a href="dashboard.html" class="btn btn-secondary fw-bold">Regresar</a>
                </div>
            </div>
        `;
    }
}

// =======================================
// 9. INICIALIZACIÓN AUTOMÁTICA
// =======================================
document.addEventListener("DOMContentLoaded", () => {
    verificarRolUI();
    cargarPacientesDashboard();
    cargarCitas();
    cargarExpedientePaciente();

    // Apaga la cámara siempre que se cierre el modal SOS (botón, Esc o clic fuera)
    const sosModal = document.getElementById('sosModal');
    if (sosModal) sosModal.addEventListener('hidden.bs.modal', detenerCamaraSOS);

    // Precarga los modelos solo en las páginas que los usan (no en patient.html)
    if (sosModal || document.getElementById('estadoModeloRegistro')) {
        inicializarModelosIA();
    }
});