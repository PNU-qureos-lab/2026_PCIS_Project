(function () {
  const palettes = {
    depth_m: {colors:['#30123b','#4145ab','#2aa7d6','#39d353','#f4e61e','#f47918','#7a0403'], labels:['0 m','3','6','9','12','15 m']},
    sand_fraction: {colors:['#fff7bc','#fec44f','#8c2d04'], labels:['0','0.25','0.5','0.75','1']},
    seagrass_fraction: {colors:['#ffffcc','#78c679','#005a32'], labels:['0','0.25','0.5','0.75','1']}
  };

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('The selected map image could not be loaded.'));
      img.src = url;
    });
  }

  function text(ctx, value, x, y, size=20, weight='normal', color='#17384c') {
    ctx.fillStyle = color;
    ctx.font = `${weight} ${size}px Arial`;
    ctx.fillText(value, x, y);
  }

  function gradientLegend(ctx, kind, x, y, width) {
    const spec = palettes[kind];
    const gradient = ctx.createLinearGradient(x, 0, x + width, 0);
    spec.colors.forEach((color, i) => gradient.addColorStop(i / (spec.colors.length - 1), color));
    ctx.fillStyle = gradient;
    ctx.fillRect(x, y, width, 24);
    ctx.strokeStyle = '#6f8490';
    ctx.strokeRect(x, y, width, 24);
    ctx.textAlign = 'center';
    spec.labels.forEach((label, i) => text(ctx, label, x + i * width / (spec.labels.length - 1), y + 51, 16));
    ctx.textAlign = 'left';
    if (kind === 'depth_m') {
      text(ctx, 'Black: 15 m or deeper. Transparent: outside model mask.', x, y + 82, 15, 'normal', '#526e7e');
    } else {
      ctx.fillStyle = '#de28aa'; ctx.fillRect(x, y + 67, 17, 17);
      text(ctx, 'Below 0', x + 25, y + 82, 15);
      ctx.fillStyle = '#1769ff'; ctx.fillRect(x + 118, y + 67, 17, 17);
      text(ctx, 'Above 1', x + 143, y + 82, 15);
      text(ctx, 'Transparent: missing or invalid.', x, y + 108, 15, 'normal', '#526e7e');
    }
  }

  function habitatLegend(ctx, x, y) {
    const rows = [['#edc15b','Sand ≥70%'],['#16894b','Seagrass ≥70%'],['#893faf','Mixed'],['#87949a','Unclassified']];
    rows.forEach((row, i) => {
      ctx.fillStyle = row[0]; ctx.fillRect(x, y + i * 34, 20, 20);
      ctx.strokeStyle = '#6f8490'; ctx.strokeRect(x, y + i * 34, 20, 20);
      text(ctx, row[1], x + 31, y + 17 + i * 34, 17);
    });
    text(ctx, 'Transparent: NoData.', x, y + 153, 15, 'normal', '#526e7e');
  }

  window.saveResultMap = async function ({imageUrl, scene, product, productLabel}) {
    const img = await loadImage(imageUrl);
    const side = 390, top = 86, pad = 30;
    const scale = Math.min(1, 1600 / img.naturalWidth);
    const mapWidth = Math.round(img.naturalWidth * scale);
    const mapHeight = Math.round(img.naturalHeight * scale);
    const canvas = document.createElement('canvas');
    canvas.width = mapWidth + side + pad * 3;
    canvas.height = Math.max(mapHeight + top + pad, 420);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    text(ctx, `${scene} · ${productLabel}`, pad, 43, 27, 'bold');
    text(ctx, 'Sentinel-2 semi-analytical result', pad, 70, 17, 'normal', '#587283');
    ctx.drawImage(img, pad, top, mapWidth, mapHeight);
    const x = pad * 2 + mapWidth;
    text(ctx, `${productLabel} legend`, x, top + 20, 21, 'bold');
    if (product === 'habitat') habitatLegend(ctx, x, top + 48);
    else gradientLegend(ctx, product, x, top + 48, side - pad);
    text(ctx, 'CRS: EPSG:32748 · 20 m pixels', x, canvas.height - 52, 15, 'normal', '#587283');
    text(ctx, '2026 PCIS Project', x, canvas.height - 27, 15, 'bold', '#587283');
    const link = document.createElement('a');
    link.download = `${scene}_${product}_map_with_legend.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };
})();
