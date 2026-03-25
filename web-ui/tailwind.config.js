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
      },
      colors: {
        surface: {
          DEFAULT: '#2B2520',
          raised: '#342E28',
          overlay: '#3D3630',
          sunken: '#231E1A',
          border: '#4A423A',
          'border-light': '#5A514A',
        },
        sand: {
          50: '#FAF8F5',
          100: '#F0EBE4',
          200: '#E0D8CD',
          300: '#C8BFB2',
          400: '#A89E91',
          500: '#8A7F72',
          600: '#6B6158',
          700: '#524A42',
          800: '#3D3630',
          900: '#2B2520',
        },
        accent: {
          DEFAULT: '#D4845A',
          light: '#E09A74',
          dark: '#B86D42',
        },
        text: {
          primary: '#F0EBE4',
          secondary: '#A89E91',
          tertiary: '#8A7F72',
          muted: '#6B6158',
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
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
      },
    },
  },
  plugins: [],
}
