import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "Hydrogen Atom — Orbital Visualizer" },
      {
        name: "description",
        content:
          "Interactive 3D visualization of hydrogen atomic orbital probability densities.",
      },
    ],
  }),
});

// ---------- Hydrogen wavefunction math ----------
// Associated Laguerre polynomial L_{n}^{alpha}(x) via recurrence
function assocLaguerre(n: number, alpha: number, x: number): number {
  if (n === 0) return 1;
  if (n === 1) return 1 + alpha - x;
  let Lkm1 = 1;
  let Lk = 1 + alpha - x;
  for (let k = 1; k < n; k++) {
    const Lkp1 = ((2 * k + 1 + alpha - x) * Lk - (k + alpha) * Lkm1) / (k + 1);
    Lkm1 = Lk;
    Lk = Lkp1;
  }
  return Lk;
}

// Associated Legendre P_l^m(x) for m >= 0
function assocLegendre(l: number, m: number, x: number): number {
  let pmm = 1;
  if (m > 0) {
    const somx2 = Math.sqrt(Math.max(0, 1 - x * x));
    let fact = 1;
    for (let i = 1; i <= m; i++) {
      pmm *= -fact * somx2;
      fact += 2;
    }
  }
  if (l === m) return pmm;
  let pmmp1 = x * (2 * m + 1) * pmm;
  if (l === m + 1) return pmmp1;
  let pll = 0;
  for (let ll = m + 2; ll <= l; ll++) {
    pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;
}

function factorial(n: number): number {
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

// |psi_{nlm}|^2 (unnormalized-friendly; absolute normalization not needed for sampling ratios)
function probDensity(n: number, l: number, m: number, r: number, theta: number): number {
  const am = Math.abs(m);
  // Radial part (Bohr radius = 1)
  const rho = (2 * r) / n;
  const radialNorm = Math.sqrt(
    Math.pow(2 / n, 3) * factorial(n - l - 1) / (2 * n * Math.pow(factorial(n + l), 3)),
  );
  const R =
    radialNorm *
    Math.exp(-rho / 2) *
    Math.pow(rho, l) *
    assocLaguerre(n - l - 1, 2 * l + 1, rho);

  // Angular part magnitude (phi dependence is |e^{i m phi}|^2 = 1)
  const cosT = Math.cos(theta);
  const legendre = assocLegendre(l, am, cosT);
  const angNorm = Math.sqrt(
    ((2 * l + 1) / (4 * Math.PI)) * (factorial(l - am) / factorial(l + am)),
  );
  const Y = angNorm * legendre;

  return R * R * Y * Y;
}

function Index() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  const clipRef = useRef({ x: 1, y: 1, z: 1 });

  const [n, setN] = useState(4);
  const [l, setL] = useState(3);
  const [mQ, setMQ] = useState(1);
  const [intensity, setIntensity] = useState(1);
  const [clipX, setClipX] = useState(1);
  const [clipY, setClipY] = useState(1);
  const [clipZ, setClipZ] = useState(1);

  // Constrain l < n, |m| <= l
  useEffect(() => {
    if (l > n - 1) setL(n - 1);
  }, [n, l]);
  useEffect(() => {
    if (Math.abs(mQ) > l) setMQ(Math.max(-l, Math.min(l, mQ)));
  }, [l, mQ]);

  // Set up scene once
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(
      50,
      mount.clientWidth / mount.clientHeight,
      0.1,
      2000,
    );
    camera.position.set(0, 0, 45);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Simple orbit controls (mouse drag + wheel zoom)
    let isDragging = false;
    let lastX = 0;
    let lastY = 0;
    let yaw = 0;
    let pitch = 0;
    let distance = 45;

    const el = renderer.domElement;
    el.style.cursor = "grab";
    const onDown = (e: PointerEvent) => {
      isDragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      el.style.cursor = "grabbing";
      el.setPointerCapture(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      yaw -= dx * 0.005;
      pitch -= dy * 0.005;
      pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, pitch));
    };
    const onUp = (e: PointerEvent) => {
      isDragging = false;
      el.style.cursor = "grab";
      try {
        el.releasePointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      distance *= 1 + e.deltaY * 0.001;
      distance = Math.max(10, Math.min(300, distance));
    };
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
    el.addEventListener("wheel", onWheel, { passive: false });

    const material = new THREE.ShaderMaterial({
      uniforms: {
        uIntensity: { value: 1.0 },
        uClip: { value: new THREE.Vector3(1, 1, 1) },
        uSize: { value: 0.3 },
      },
      vertexShader: `
        attribute float aProb;
        varying float vProb;
        varying vec3 vPos;
        uniform float uSize;
        void main() {
          vProb = aProb;
          vPos = position;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = uSize * (320.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        varying float vProb;
        varying vec3 vPos;
        uniform float uIntensity;
        uniform vec3 uClip;

        // Plasma-style palette: deep purple -> magenta -> red -> orange -> yellow
        vec3 palette(float t) {
          vec3 c0 = vec3(0.00, 0.00, 0.18); // very dark navy
          vec3 c1 = vec3(0.00, 0.05, 0.55); // deep blue
          vec3 c2 = vec3(0.00, 0.55, 0.85); // cyan
          vec3 c3 = vec3(0.95, 0.90, 0.05); // bright yellow
          vec3 c4 = vec3(0.85, 0.05, 0.00); // red
          vec3 col;
          if (t < 0.30)      col = mix(c0, c1, t / 0.30);
          else if (t < 0.60) col = mix(c1, c2, (t - 0.30) / 0.30);
          else if (t < 0.85) col = mix(c2, c3, (t - 0.60) / 0.25);
          else               col = mix(c3, c4, (t - 0.85) / 0.15);
          return col;
        }

        void main() {
          vec2 c = gl_PointCoord - 0.5;
          float d = length(c);
          if (d > 0.5) discard;

          float extent = 20.0;
          if (abs(vPos.x) > uClip.x * extent) discard;
          if (abs(vPos.y) > uClip.y * extent) discard;
          if (abs(vPos.z) > uClip.z * extent) discard;

          float p = clamp(pow(vProb, 0.5) * uIntensity, 0.0, 1.0);
          vec3 col = palette(p);

          // Subtle sphere-like shading so tiny points remain distinct dots
          vec2 lightDir = vec2(-0.35, 0.35);
          float nz = sqrt(max(0.0, 1.0 - 4.0 * d * d));
          float lambert = clamp(nz * 0.35 + dot(-c * 2.0, lightDir) * 0.12 + 0.65, 0.0, 1.0);
          col *= lambert;

          // Hard round edge (tiny sphere look), no additive halo
          float alpha = smoothstep(0.5, 0.42, d);
          gl_FragColor = vec4(col, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
    });


    materialRef.current = material;

    let raf = 0;
    const animate = () => {
      raf = requestAnimationFrame(animate);
      const cx = distance * Math.cos(pitch) * Math.sin(yaw);
      const cy = distance * Math.sin(pitch);
      const cz = distance * Math.cos(pitch) * Math.cos(yaw);
      camera.position.set(cx, cy, cz);
      camera.lookAt(0, 0, 0);
      material.uniforms.uClip.value.set(clipRef.current.x, clipRef.current.y, clipRef.current.z);
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      if (!mount) return;
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    // Store scene for sampling effect
    (mount as unknown as { __scene: THREE.Scene }).__scene = scene;

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
      el.removeEventListener("wheel", onWheel);
      renderer.dispose();
      material.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };
  }, []);

  // Regenerate point cloud when quantum numbers change
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const scene = (mount as unknown as { __scene?: THREE.Scene }).__scene;
    const material = materialRef.current;
    if (!scene || !material) return;

    // Remove old
    if (pointsRef.current) {
      scene.remove(pointsRef.current);
      pointsRef.current.geometry.dispose();
      pointsRef.current = null;
    }

    // Rejection sampling
    const target = 500000;
    const maxTries = target * 40;

    // Estimate rMax scales with n^2
    const rMax = 2.2 * n * n + 4;
    // Estimate peak density by sampling a coarse grid
    let peak = 0;
    const gridR = 30;
    const gridT = 30;
    for (let i = 1; i <= gridR; i++) {
      const r = (i / gridR) * rMax;
      for (let j = 0; j < gridT; j++) {
        const t = (j / (gridT - 1)) * Math.PI;
        const p = probDensity(n, l, mQ, r, t) * r * r * Math.sin(t);
        if (p > peak) peak = p;
      }
    }
    if (peak <= 0) peak = 1;

    const positions = new Float32Array(target * 3);
    const probs = new Float32Array(target);
    let accepted = 0;
    let tries = 0;
    // Scale so that Bohr-radius units map to a nice view volume
    const scale = 20 / rMax;

    while (accepted < target && tries < maxTries) {
      tries++;
      const r = Math.random() * rMax;
      const cosT = 2 * Math.random() - 1;
      const theta = Math.acos(cosT);
      const phi = Math.random() * 2 * Math.PI;
      const p = probDensity(n, l, mQ, r, theta) * r * r * Math.sin(theta);
      if (Math.random() * peak < p) {
        const sinT = Math.sin(theta);
        const x = r * sinT * Math.cos(phi) * scale;
        const y = r * cosT * scale;
        const z = r * sinT * Math.sin(phi) * scale;
        positions[accepted * 3] = x;
        positions[accepted * 3 + 1] = y;
        positions[accepted * 3 + 2] = z;
        probs[accepted] = Math.min(1, p / peak);
        accepted++;
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions.slice(0, accepted * 3), 3));
    geometry.setAttribute("aProb", new THREE.BufferAttribute(probs.slice(0, accepted), 1));

    const pts = new THREE.Points(geometry, material);
    scene.add(pts);
    pointsRef.current = pts;
  }, [n, l, mQ]);

  // Push intensity + clip uniforms
  useEffect(() => {
    if (materialRef.current) materialRef.current.uniforms.uIntensity.value = intensity;
  }, [intensity]);
  useEffect(() => {
    clipRef.current = { x: clipX, y: clipY, z: clipZ };
  }, [clipX, clipY, clipZ]);

  return (
    <div className="min-h-screen w-full bg-black text-zinc-300 font-mono">
      <div className="mx-auto w-full max-w-[800px] px-4 pt-12 pb-16">
        <header className="mb-6">
          <h1 className="text-4xl tracking-tight text-white">hydrogen atom</h1>
          <p className="mt-1 text-xs text-zinc-500">
            Orbital probability density visualizer
          </p>
          <div className="mt-4 h-px w-full bg-zinc-800" />
        </header>

        <div className="mt-6 flex justify-center">
          <div className="flex w-full flex-col gap-4">
            <div className="relative h-[500px] w-full overflow-hidden rounded-xl border border-zinc-800 bg-black shadow-2xl md:h-[600px]">
              <div ref={mountRef} className="h-full w-full" />
              <div className="pointer-events-none absolute bottom-3 left-4 text-[10px] text-white/50">
                HYDROGEN ATOM · ORBITAL PROBABILITY DENSITY
              </div>
              <div className="pointer-events-none absolute top-3 right-4 text-[10px] text-white/40">
                n={n} &nbsp; l={l} &nbsp; m={mQ}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-8 rounded-xl border border-zinc-800 bg-zinc-900/50 p-6 md:grid-cols-2">
              <div className="flex flex-col gap-4">
                <h3 className="text-[10px] tracking-widest text-zinc-500">QUANTUM NUMBERS</h3>
                <Slider label="n (Principal)" value={n} min={1} max={7} onChange={setN} />
                <Slider label="l (Azimuthal)" value={l} min={0} max={Math.max(0, n - 1)} onChange={setL} />
                <Slider label="m (Magnetic)" value={mQ} min={-l} max={l} onChange={setMQ} />
              </div>
              <div className="flex flex-col gap-4">
                <h3 className="text-[10px] tracking-widest text-zinc-500">VISUALIZATION</h3>
                <Slider
                  label="Color Intensity"
                  value={intensity}
                  min={0.1}
                  max={3}
                  step={0.1}
                  onChange={setIntensity}
                  format={(v) => v.toFixed(1)}
                />
                <div>
                  <p className="mb-2 text-[10px] tracking-widest text-zinc-500">PLANE CLIPPING</p>
                  <div className="grid grid-cols-3 gap-3">
                    <Slider label="X" value={clipX} min={0} max={1} step={0.01} onChange={setClipX} compact format={(v) => v.toFixed(2)} />
                    <Slider label="Y" value={clipY} min={0} max={1} step={0.01} onChange={setClipY} compact format={(v) => v.toFixed(2)} />
                    <Slider label="Z" value={clipZ} min={0} max={1} step={0.01} onChange={setClipZ} compact format={(v) => v.toFixed(2)} />
                  </div>
                </div>
              </div>
            </div>

            <p className="mt-2 text-center text-[10px] text-zinc-600">
              Drag to rotate · scroll to zoom
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  compact,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  compact?: boolean;
  format?: (v: number) => string;
}) {
  return (
    <div className={compact ? "flex flex-col gap-1" : "flex flex-col gap-1"}>
      <div className="flex items-center justify-between text-[11px] text-zinc-400">
        <span>{label}</span>
        <span className="text-zinc-500">{format ? format(value) : value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="accent-white"
      />
    </div>
  );
}
