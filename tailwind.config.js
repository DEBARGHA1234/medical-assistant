/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0F172A",
        surface: "#1E293B",
        primary: "#0EA5E9",
        accent: "#10B981",
        warning: "#F59E0B",
        danger: "#EF4444",
        textLight: "#F1F5F9",
        textDark: "#94A3B8"
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"]
      }
    },
  },
  plugins: [],
}
