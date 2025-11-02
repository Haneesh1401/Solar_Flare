import React, { useEffect, useRef } from 'react';

function Starfield({ density = 200, speed = 0.05 }) {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const starsRef = useRef([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const createStars = () => {
      starsRef.current = Array.from({ length: density }).map(() => ({
        x: Math.random() * width,
        y: Math.random() * height,
        z: Math.random() * 0.7 + 0.3,
        r: Math.random() * 1.5 + 0.2
      }));
    };

    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      for (const star of starsRef.current) {
        const glow = 0.2 + 0.8 * star.z;
        ctx.beginPath();
        ctx.fillStyle = `rgba(255,255,255,${glow})`;
        ctx.shadowColor = 'rgba(0,255,255,0.3)';
        ctx.shadowBlur = 8 * star.z;
        ctx.arc(star.x, star.y, star.r * star.z, 0, Math.PI * 2);
        ctx.fill();

        star.y += speed * (0.5 + star.z);
        if (star.y > height) {
          star.y = -2;
          star.x = Math.random() * width;
        }
      }
    };

    const loop = () => {
      draw();
      animationRef.current = requestAnimationFrame(loop);
    };

    const onResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      createStars();
    };

    createStars();
    loop();
    window.addEventListener('resize', onResize);
    return () => {
      cancelAnimationFrame(animationRef.current);
      window.removeEventListener('resize', onResize);
    };
  }, [density, speed]);

  return <canvas ref={canvasRef} className="starfield-canvas" aria-hidden="true" />;
}

export default Starfield;


