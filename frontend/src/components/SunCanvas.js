import React, { Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';

function SunSphere({ flareIntensity = 0.4 }) {
  const color = useMemo(() => {
    if (flareIntensity > 0.8) return '#ff3b3b';
    if (flareIntensity > 0.5) return '#ff9f1c';
    return '#ffd166';
  }, [flareIntensity]);

  return (
    <group>
      <mesh>
        <sphereGeometry args={[2.2, 64, 64]} />
        <meshStandardMaterial emissive={color} emissiveIntensity={1.5} color="#111" />
      </mesh>
      <mesh>
        <sphereGeometry args={[2.4, 64, 64]} />
        <meshBasicMaterial color={color} transparent opacity={0.25} />
      </mesh>
    </group>
  );
}

function SunCanvas({ intensity = 0.4 }) {
  return (
    <div className="w-full h-72 md:h-96 rounded-lg overflow-hidden border border-accent-orange/30 shadow-neon" role="img" aria-label="3D Sun visualization showing solar activity">
      <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
        <ambientLight intensity={0.2} />
        <pointLight position={[6, 6, 6]} intensity={2.5} color="#ffbf69" />
        <Suspense fallback={null}>
          <SunSphere flareIntensity={intensity} />
        </Suspense>
        <Stars radius={50} depth={20} count={1000} factor={2} saturation={0} fade speed={1} />
        <OrbitControls enablePan={false} enableZoom={false} autoRotate autoRotateSpeed={0.7} />
      </Canvas>
    </div>
  );
}

export default SunCanvas;


