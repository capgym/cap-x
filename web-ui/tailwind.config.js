/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        surface: {
          DEFAULT: '#0a0a0a',
          raised: '#141414',
          overlay: '#1a1a1a',
          sunken: '#050505',
          border: '#262626',
          'border-light': '#333333',
        },
        sand: {
          50: '#fafafa',
          100: '#f0f0f0',
          200: '#d4d4d4',
          300: '#a3a3a3',
          400: '#737373',
          500: '#525252',
          600: '#404040',
          700: '#2a2a2a',
          800: '#1a1a1a',
          900: '#0a0a0a',
        },
        accent: {
          DEFAULT: '#D4A017',
          light: '#F0C040',
          dark: '#B8860B',
        },
        nv: {
          green: '#76b900',
          'green-light': '#8ecf10',
        },
        text: {
          primary: '#E8E8E8',
          secondary: '#888888',
          tertiary: '#737373',
          muted: '#404040',
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'shimmer': 'shimmer 2s ease-in-out infinite',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%, 100%': { opacity: '0.5' },
          '50%': { opacity: '1' },
        },
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 8px 0 rgba(212, 160, 23, 0.15)' },
          '50%': { boxShadow: '0 0 16px 2px rgba(212, 160, 23, 0.3)' },
        },
      },
    },
  },
  plugins: [],
}
