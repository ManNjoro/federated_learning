// tailwind.config.js
export const content = ['./index.html', './src/**/*.{js,ts,jsx,tsx}'];
export const theme = {
    extend: {
        colors: {
            medical: {
                blue: '#1a365d',
                teal: '#2c7a7b',
                alert: '#c53030'
            }
        }
    },
};
export const plugins = [];