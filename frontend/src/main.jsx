import React from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/bricolage-grotesque/500.css";
import "@fontsource/bricolage-grotesque/700.css";
import "@fontsource/atkinson-hyperlegible/400.css";
import "@fontsource/atkinson-hyperlegible/700.css";
import "./estilos.css";
import App from "./App.jsx";

createRoot(document.getElementById("raiz")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
