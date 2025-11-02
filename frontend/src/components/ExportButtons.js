import React from 'react';
import { toPng } from 'html-to-image';
import jsPDF from 'jspdf';

function ExportButtons({ targetId = 'dashboard-root' }) {
  const handlePNG = async () => {
    const node = document.getElementById(targetId);
    if (!node) return;
    const dataUrl = await toPng(node, { cacheBust: true, pixelRatio: 2 });
    const link = document.createElement('a');
    link.download = 'solar-dashboard.png';
    link.href = dataUrl;
    link.click();
  };

  const handlePDF = async () => {
    const node = document.getElementById(targetId);
    if (!node) return;
    const dataUrl = await toPng(node, { cacheBust: true, pixelRatio: 2 });
    const pdf = new jsPDF('l', 'pt', 'a4');
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    pdf.addImage(dataUrl, 'PNG', 0, 0, pageWidth, pageHeight);
    pdf.save('solar-dashboard.pdf');
  };

  return (
    <div className="flex gap-2">
      <button className="px-3 py-2 rounded bg-accent-cyan text-space-dark font-bold" onClick={handlePNG} aria-label="Export dashboard as PNG">
        Export PNG
      </button>
      <button className="px-3 py-2 rounded bg-accent-orange text-space-dark font-bold" onClick={handlePDF} aria-label="Export dashboard as PDF">
        Export PDF
      </button>
    </div>
  );
}

export default ExportButtons;


