import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Features from './pages/Features';
import PredictHeart from './pages/PredictHeart';
import PredictDiabetes from './pages/PredictDiabetes';
import Findings from './pages/Findings';
import History from './pages/History';
import ModelComparison from './pages/ModelComparison';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route path="/features" element={<Features />} />
          <Route path="/predict/heart" element={<PredictHeart />} />
          <Route path="/predict/diabetes" element={<PredictDiabetes />} />
          <Route path="/findings" element={<Findings />} />
          <Route path="/models" element={<ModelComparison />} />
          <Route path="/history" element={<History />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
