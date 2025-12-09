import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const container = document.getElementById('app');
const statusEl = document.getElementById('status');
const pointSizeSlider = document.getElementById('point-size');
const toggleAutorotate = document.getElementById('toggle-autorotate');
const toggleGrid = document.getElementById('toggle-grid');

let renderer;
let scene;
let camera;
let controls;
let gridHelper;
let currentPoints;
let cameraMarkers = [];
let tourPoses = [];
let tourPlaying = false;
let markerClickAttached = false;

// Keyboard controls
const keyboard = {
    w: false, a: false, s: false, d: false, // Movement
    r: false, f: false, // Up/Down
    ArrowUp: false, ArrowDown: false, ArrowLeft: false, ArrowRight: false // Rotation
};
const moveSpeed = 0.05;
const rotateSpeed = 0.02;

const loader = new PLYLoader();
const toastContainer = document.getElementById('toast-container');

console.log('PLYLoader instance:', loader);
console.log('Initializing Three.js viewer...');
init();
attachUI();
render();

function init() {
    console.log('Creating renderer and scene...');
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x0c1018, 1);
    container.appendChild(renderer.domElement);

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 2000);
    camera.position.set(2.4, 1.2, 2.4);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = false;

    const ambient = new THREE.AmbientLight(0xffffff, 0.35);
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(4, 6, 4);
    scene.add(ambient, dir);

    gridHelper = new THREE.GridHelper(20, 40, 0x2e3c4f, 0x1b2533);
    gridHelper.material.opacity = 0.35;
    gridHelper.material.transparent = true;
    scene.add(gridHelper);
    console.log('Scene initialized. Camera at:', camera.position);

    window.addEventListener('resize', onWindowResize);
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', handleDrop);

    // Keyboard controls
    window.addEventListener('keydown', (e) => {
        const key = e.key.toLowerCase();
        if (key in keyboard || e.key in keyboard) {
            keyboard[key] || (keyboard[e.key] = true);
            if (key === 'w' || key === 'a' || key === 's' || key === 'd' ||
                key === 'r' || key === 'f' || e.key.startsWith('Arrow')) {
                keyboard[key in keyboard ? key : e.key] = true;
            }
        }
    });

    window.addEventListener('keyup', (e) => {
        const key = e.key.toLowerCase();
        if (key in keyboard || e.key in keyboard) {
            keyboard[key in keyboard ? key : e.key] = false;
        }
    });
}

function attachUI() {
    document.getElementById('file-input').addEventListener('change', (e) => {
        const file = e.target.files?.[0];
        if (file) loadPLYFromFile(file);
    });

    document.getElementById('btn-load-url').addEventListener('click', () => {
        const url = document.getElementById('url-input').value.trim();
        if (!url) return;
        loadPLYFromUrl(url);
    });

    document.getElementById('btn-sample').addEventListener('click', () => {
        console.log('Sample Cloud button clicked');
        loadPLYFromUrl('./data/meshlab.ply');
    });

    document.getElementById('btn-screenshot').addEventListener('click', saveScreenshot);

    pointSizeSlider.addEventListener('input', () => {
        if (currentPoints) {
            currentPoints.material.size = parseFloat(pointSizeSlider.value);
        }
    });

    toggleAutorotate.addEventListener('change', () => {
        controls.autoRotate = toggleAutorotate.checked;
    });

    toggleGrid.addEventListener('change', () => {
        gridHelper.visible = toggleGrid.checked;
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        resetView();
    });

    document.getElementById('pose-input').addEventListener('change', (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        loadPosesFromFile(file);
    });

    document.getElementById('btn-sample-poses').addEventListener('click', () => {
        loadPosesFromUrl('./data/meshlab_poses.json');
    });

    document.getElementById('btn-play').addEventListener('click', () => {
        if (tourPoses.length === 0) {
            setStatus('Load camera poses to enable the tour.');
            return;
        }
        playTour();
    });

    document.getElementById('btn-stop').addEventListener('click', stopTour);

    const dropzone = document.getElementById('dropzone');
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('active');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('active'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('active');
        const file = e.dataTransfer?.files?.[0];
        if (file && file.name.toLowerCase().endsWith('.ply')) {
            loadPLYFromFile(file);
        }
    });
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function render() {
    requestAnimationFrame(render);

    // Process keyboard controls
    if (keyboard.w || keyboard.a || keyboard.s || keyboard.d ||
        keyboard.r || keyboard.f ||
        keyboard.ArrowUp || keyboard.ArrowDown || keyboard.ArrowLeft || keyboard.ArrowRight) {

        // Temporarily disable auto-rotate when using keyboard
        const wasAutoRotating = controls.autoRotate;
        controls.autoRotate = false;

        // Get camera direction vectors
        const forward = new THREE.Vector3();
        camera.getWorldDirection(forward);
        forward.y = 0; // Keep movement horizontal
        forward.normalize();

        const right = new THREE.Vector3();
        right.crossVectors(forward, camera.up).normalize();

        // Movement (WASD)
        if (keyboard.w) {
            camera.position.addScaledVector(forward, moveSpeed);
            controls.target.addScaledVector(forward, moveSpeed);
        }
        if (keyboard.s) {
            camera.position.addScaledVector(forward, -moveSpeed);
            controls.target.addScaledVector(forward, -moveSpeed);
        }
        if (keyboard.a) {
            camera.position.addScaledVector(right, -moveSpeed);
            controls.target.addScaledVector(right, -moveSpeed);
        }
        if (keyboard.d) {
            camera.position.addScaledVector(right, moveSpeed);
            controls.target.addScaledVector(right, moveSpeed);
        }

        // Vertical movement (R/F)
        if (keyboard.r) {
            camera.position.y += moveSpeed;
            controls.target.y += moveSpeed;
        }
        if (keyboard.f) {
            camera.position.y -= moveSpeed;
            controls.target.y -= moveSpeed;
        }

        // Rotation (Arrow keys)
        const toTarget = new THREE.Vector3().subVectors(controls.target, camera.position);
        const distance = toTarget.length();

        if (keyboard.ArrowLeft || keyboard.ArrowRight) {
            const angle = keyboard.ArrowLeft ? rotateSpeed : -rotateSpeed;
            const axis = new THREE.Vector3(0, 1, 0);
            toTarget.applyAxisAngle(axis, angle);
            controls.target.copy(camera.position).add(toTarget);
        }

        if (keyboard.ArrowUp || keyboard.ArrowDown) {
            const angle = keyboard.ArrowUp ? rotateSpeed : -rotateSpeed;
            toTarget.normalize();

            // Calculate current vertical angle
            const currentVerticalAngle = Math.asin(toTarget.y);
            const maxVerticalAngle = Math.PI / 2 - 0.1; // Limit to 89 degrees (prevent gimbal lock)

            // Check if rotation would exceed limits
            const newVerticalAngle = currentVerticalAngle + angle;

            if (Math.abs(newVerticalAngle) < maxVerticalAngle) {
                // Rotate around the right vector
                const currentRight = new THREE.Vector3();
                currentRight.crossVectors(toTarget, camera.up).normalize();
                toTarget.applyAxisAngle(currentRight, angle);

                toTarget.multiplyScalar(distance);
                controls.target.copy(camera.position).add(toTarget);
            }
        }

        controls.update();

        // Restore auto-rotate state
        controls.autoRotate = wasAutoRotating;
    } else {
        controls.update();
    }

    renderer.render(scene, camera);
}

async function loadPLYFromFile(file) {
    setStatus(`Loading ${file.name} ...`);
    showToast(`Loading ${file.name}...`, 'info', 2000);
    const buffer = await file.arrayBuffer();
    parsePLY(buffer, file.name);
}

async function loadPLYFromUrl(url) {
    console.log('loadPLYFromUrl called with:', url);
    try {
        setStatus(`Fetching ${url} ...`);
        showToast(`Fetching point cloud...`, 'info', 2000);
        console.log('Starting fetch...');
        const res = await fetch(url);
        console.log('Fetch response:', res.status, res.statusText);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        console.log('Reading buffer...');
        const buffer = await res.arrayBuffer();
        console.log('Buffer size:', buffer.byteLength, 'bytes');
        parsePLY(buffer, url);
    } catch (err) {
        console.error('❌ Load error:', err);
        setStatus(`Failed to load PLY: ${err}`);
        showToast(`Failed to load PLY: ${err.message}`, 'error', 5000);
    }
}

function parsePLY(buffer, label) {
    console.log('====== parsePLY CALLED ======');
    console.log('Buffer byteLength:', buffer.byteLength);
    console.log('Buffer type:', buffer.constructor.name);
    console.log('Label:', label);
    console.log('Loader exists:', !!loader);
    console.log('Loader type:', loader?.constructor?.name);
    console.log('Loader.parse is function:', typeof loader?.parse === 'function');

    // Try to see first few bytes
    const view = new Uint8Array(buffer.slice(0, 20));
    const text = new TextDecoder().decode(view);
    console.log('First 20 bytes as text:', text);

    try {
        console.log('Calling loader.parse...');
        // PLYLoader.parse() returns the geometry synchronously
        const geometry = loader.parse(buffer);
        console.log('✓ PLY parsed successfully');
        console.log('Geometry:', geometry);
        console.log('Position count:', geometry.attributes.position?.count || 0);
        console.log('Has colors:', geometry.hasAttribute('color'));

        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        console.log('Bounding sphere:', geometry.boundingSphere);

        const hasColor = geometry.hasAttribute('color');
        const material = new THREE.PointsMaterial({
            size: parseFloat(pointSizeSlider.value),
            sizeAttenuation: true,
            vertexColors: hasColor,
            color: hasColor ? 0xffffff : 0x55d6be,
            opacity: 1.0,
            transparent: false
        });

        if (currentPoints) {
            scene.remove(currentPoints);
            currentPoints.geometry.dispose();
            currentPoints.material.dispose();
        }

        currentPoints = new THREE.Points(geometry, material);
        scene.add(currentPoints);
        console.log('✓ Points added to scene. Scene children:', scene.children.length);
        console.log('Point cloud visible:', currentPoints.visible);
        console.log('Point cloud position:', currentPoints.position);
        console.log('Point size:', material.size);

        fitViewToObject(currentPoints, 1.4);
        const pointCount = geometry.attributes.position.count;
        setStatus(`Loaded ${label} (${pointCount} points)`);
        showToast(`✓ Point cloud loaded: ${pointCount.toLocaleString()} points`, 'success', 4000);
        console.log('loader.parse() completed successfully');
    } catch (err) {
        console.error('PLY loading error:', err);
        setStatus(`Error loading PLY: ${err.message}`);
        showToast(`Error loading PLY: ${err.message}`, 'error', 5000);
    }
}

function fitViewToObject(object, pad = 1.2) {
    const box = new THREE.Box3().setFromObject(object);
    if (!box.isEmpty()) {
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        const maxDim = Math.max(size.x, size.y, size.z);
        const distance = (maxDim / Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5))) * pad;

        const dir = new THREE.Vector3(1, 0.8, 1).normalize();
        const newPos = center.clone().add(dir.multiplyScalar(distance));

        camera.position.copy(newPos);
        controls.target.copy(center);
        controls.update();
    }
}

function resetView() {
    controls.autoRotate = false;
    toggleAutorotate.checked = false;
    controls.reset();
    camera.position.set(2.4, 1.2, 2.4);
    controls.target.set(0, 0, 0);
    controls.update();
}

function saveScreenshot() {
    renderer.render(scene, camera);
    const url = renderer.domElement.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = 'virtual-tour.png';
    a.click();
    showToast('✓ Screenshot saved', 'success', 2500);
}

async function loadPosesFromFile(file) {
    const text = await file.text();
    try {
        const json = JSON.parse(text);
        applyPoses(json, file.name);
    } catch (err) {
        setStatus('Pose JSON parse error.');
    }
}

async function loadPosesFromUrl(url) {
    try {
        const res = await fetch(url);
        const json = await res.json();
        applyPoses(json, url);
    } catch (err) {
        setStatus(`Failed to load poses: ${err}`);
    }
}

function applyPoses(json, label) {
    const cams = json.cameras || json.poses || [];
    const parsed = cams
        .map((c) => convertPose(c))
        .filter(Boolean);

    if (parsed.length === 0) {
        setStatus('No valid poses found. Expect position [x,y,z].');
        showToast('No valid poses found in JSON', 'error', 4000);
        return;
    }

    tourPoses = parsed;
    placeCameraMarkers(parsed);
    setStatus(`Loaded ${parsed.length} camera poses from ${label}`);
    showToast(`✓ Loaded ${parsed.length} camera viewpoints`, 'success', 3000);
}

function convertPose(pose) {
    if (!pose) return null;
    let position = null;
    if (Array.isArray(pose.position) && pose.position.length === 3) {
        position = new THREE.Vector3().fromArray(pose.position);
    } else if (pose.R && pose.t) {
        try {
            const R = new THREE.Matrix3().set(
                pose.R[0][0], pose.R[0][1], pose.R[0][2],
                pose.R[1][0], pose.R[1][1], pose.R[1][2],
                pose.R[2][0], pose.R[2][1], pose.R[2][2]
            );
            const t = new THREE.Vector3().fromArray(pose.t);
            const Rt = R.clone().transpose();
            position = t.clone().applyMatrix3(Rt).multiplyScalar(-1);
        } catch (err) {
            position = null;
        }
    }

    if (!position) return null;

    let target = new THREE.Vector3(0, 0, 0);
    if (Array.isArray(pose.target) && pose.target.length === 3) {
        target = new THREE.Vector3().fromArray(pose.target);
    }

    return {
        name: pose.name || pose.id || 'cam',
        position,
        target
    };
}

function placeCameraMarkers(poses) {
    cameraMarkers.forEach((m) => {
        scene.remove(m);
        m.geometry.dispose();
        m.material.dispose();
    });
    cameraMarkers = [];

    const baseSize = currentPoints ? new THREE.Box3().setFromObject(currentPoints).getSize(new THREE.Vector3()).length() * 0.015 : 0.05;
    const geo = new THREE.ConeGeometry(baseSize, baseSize * 2.2, 5);

    poses.forEach((pose, idx) => {
        const mat = new THREE.MeshStandardMaterial({ color: idx === 0 ? 0xffb347 : 0x55d6be, emissive: 0x072f28, metalness: 0.1, roughness: 0.5 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.copy(pose.position);
        mesh.lookAt(pose.target);
        mesh.userData.poseIndex = idx;
        mesh.cursor = 'pointer';

        mesh.onClick = () => moveCameraToPose(pose, 1.2, false);
        cameraMarkers.push(mesh);
        scene.add(mesh);
    });

    if (!markerClickAttached) {
        renderer.domElement.addEventListener('pointerdown', onPointerDownMarker);
        markerClickAttached = true;
    }
}

function onPointerDownMarker(event) {
    const mouse = new THREE.Vector2(
        (event.clientX / renderer.domElement.clientWidth) * 2 - 1,
        -(event.clientY / renderer.domElement.clientHeight) * 2 + 1
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(cameraMarkers);
    if (hits.length > 0) {
        const idx = hits[0].object.userData.poseIndex;
        const pose = tourPoses[idx];
        moveCameraToPose(pose, 1.0);
    }
}

function moveCameraToPose(pose, durationSec = 1.0, respectStop = false) {
    const startPos = camera.position.clone();
    const startTarget = controls.target.clone();
    const endPos = pose.position.clone();
    const endTarget = pose.target.clone();
    const start = performance.now();
    const durationMs = durationSec * 1000;

    return new Promise((resolve) => {
        function step(now) {
            const t = Math.min(1, (now - start) / durationMs);
            const ease = t * (2 - t); // ease out
            camera.position.lerpVectors(startPos, endPos, ease);
            controls.target.lerpVectors(startTarget, endTarget, ease);
            controls.update();
            const keepAnimating = t < 1 && (!respectStop || tourPlaying);
            if (keepAnimating) {
                requestAnimationFrame(step);
            } else {
                resolve();
            }
        }
        requestAnimationFrame(step);
    });
}

async function playTour() {
    if (tourPlaying) return;
    tourPlaying = true;
    controls.autoRotate = false;
    toggleAutorotate.checked = false;
    showToast(`▶ Starting virtual tour (${tourPoses.length} stops)`, 'info', 3000);

    for (let i = 0; i < tourPoses.length && tourPlaying; i++) {
        await moveCameraToPose(tourPoses[i], 1.2, true);
        await sleep(220);
    }

    tourPlaying = false;
    if (tourPoses.length > 0) {
        showToast('✓ Tour completed', 'success', 2500);
    }
}

function stopTour() {
    if (tourPlaying) {
        tourPlaying = false;
        showToast('Tour stopped', 'info', 2000);
    }
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function handleDrop(e) {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file && file.name.toLowerCase().endsWith('.ply')) {
        loadPLYFromFile(file);
    }
}

function setStatus(msg) {
    statusEl.textContent = msg;
}

function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icon = document.createElement('span');
    icon.className = 'toast-icon';
    icon.textContent = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';

    const messageEl = document.createElement('span');
    messageEl.className = 'toast-message';
    messageEl.textContent = message;

    toast.appendChild(icon);
    toast.appendChild(messageEl);
    toastContainer.appendChild(toast);

    let timeoutId;
    let isHovered = false;

    const dismiss = () => {
        if (isHovered) return;
        toast.classList.add('hiding');
        setTimeout(() => {
            if (toast.parentNode === toastContainer) {
                toastContainer.removeChild(toast);
            }
        }, 300);
    };

    toast.addEventListener('mouseenter', () => {
        isHovered = true;
        if (timeoutId) clearTimeout(timeoutId);
    });

    toast.addEventListener('mouseleave', () => {
        isHovered = false;
        timeoutId = setTimeout(dismiss, 1000);
    });

    toast.addEventListener('click', () => {
        if (timeoutId) clearTimeout(timeoutId);
        dismiss();
    });

    if (duration > 0) {
        timeoutId = setTimeout(dismiss, duration);
    }
}
