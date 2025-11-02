/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./public/index.html",
    "./src/**/*.{js,jsx,ts,tsx}"
  ],
  darkMode: "media",
  theme: {
    extend: {
      colors: {
        space: {
          dark: "#0A0E1A",
          light: "#1A1F2E"
        },
        accent: {
          cyan: "#00FFFF",
          orange: "#FF8C00",
          magenta: "#FF00FF",
          purple: "#9D4EDD"
        },
        text: {
          white: "#FFFFFF"
        }
      },
      fontFamily: {
        orbitron: ["Orbitron", "ui-sans-serif", "system-ui"],
        exo: ["Exo", "ui-sans-serif", "system-ui"],
        mono: ["Share Tech Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "Liberation Mono", "Courier New", "monospace"]
      },
      boxShadow: {
        glow: "0 0 20px rgba(0, 255, 255, 0.25)",
        neon: "0 0 10px rgba(255, 0, 255, 0.4), 0 0 30px rgba(0, 255, 255, 0.2)"
      },
      backgroundImage: {
        "cosmic-gradient": "linear-gradient(135deg, #0A0E1A 0%, #131C2B 100%)"
      },
      keyframes: {
        aurora: {
          '0%, 100%': { transform: 'translateX(-10%)' },
          '50%': { transform: 'translateX(10%)' }
        },
        pulseGlow: {
          '0%, 100%': { opacity: 0.5 },
          '50%': { opacity: 1 }
        }
      },
      animation: {
        aurora: 'aurora 12s ease-in-out infinite',
        pulseGlow: 'pulseGlow 3s ease-in-out infinite'
      }
    }
  },
  plugins: []
};


