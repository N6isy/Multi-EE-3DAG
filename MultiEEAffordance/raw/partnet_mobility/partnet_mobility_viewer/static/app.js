const state = {
  offset: 0,
  limit: 50,
  q: '',
  category: '',
  currentObjectId: null,
  currentDetail: null,
  three: null,
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>'"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function statCard(label, value) {
  return `<div class="stat-card"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`;
}

async function loadStats() {
  const stats = await getJson('/api/stats');
  $('stats').innerHTML = [
    statCard('zip 路径', stats.zip_path),
    statCard('object 数量', stats.object_count),
    statCard('类别数量', stats.category_count),
    statCard('mesh / json / image 文件数', `${stats.mesh_file_count} / ${stats.json_file_count} / ${stats.image_file_count}`),
  ].join('');

  const sel = $('categorySelect');
  const categories = Object.entries(stats.categories || {}).sort((a, b) => a[0].localeCompare(b[0]));
  for (const [cat, n] of categories) {
    const opt = document.createElement('option');
    opt.value = cat;
    opt.textContent = `${cat} (${n})`;
    sel.appendChild(opt);
  }
}

async function loadObjects() {
  const params = new URLSearchParams({
    q: state.q,
    category: state.category,
    limit: state.limit,
    offset: state.offset,
  });
  const data = await getJson(`/api/objects?${params.toString()}`);
  $('objectCount').textContent = `共 ${data.total} 个匹配样本；当前 ${data.offset + 1}-${Math.min(data.offset + data.limit, data.total)}`;
  const html = data.items.map(item => `
    <div class="object-card ${item.object_id === state.currentObjectId ? 'active' : ''}" data-id="${esc(item.object_id)}">
      <div class="id">${esc(item.object_id)} <span class="badge">${esc(item.category)}</span></div>
      <div class="meta">files=${item.file_count}, json=${item.json_count}, urdf=${item.urdf_count}, mesh=${item.mesh_count}, image=${item.image_count}</div>
      <div class="meta">${esc((item.key_files || []).join(' · '))}</div>
    </div>
  `).join('');
  $('objectList').innerHTML = html || '<div class="muted" style="padding:12px 0;">没有匹配样本</div>';
  document.querySelectorAll('.object-card').forEach(el => el.addEventListener('click', () => loadObject(el.dataset.id)));
  $('prevBtn').disabled = state.offset <= 0;
  $('nextBtn').disabled = state.offset + state.limit >= data.total;
}

function pretty(obj) { return JSON.stringify(obj, null, 2); }

function renderSummary(detail) {
  const s = detail.summary;
  $('summaryTab').innerHTML = `
    <div class="grid two">
      ${statCard('object_id', s.object_id)}
      ${statCard('category', s.category)}
      ${statCard('文件总数', s.file_count)}
      ${statCard('JSON / URDF / Mesh / Image', `${s.json_count} / ${s.urdf_count} / ${s.mesh_count} / ${s.image_count}`)}
    </div>
    <h3>meta.json 预览</h3>
    <pre>${esc(pretty(detail.meta_preview || '未找到 meta.json'))}</pre>
    <h3>result.json 预览</h3>
    <pre>${esc(pretty(detail.result_preview || '未找到 result.json'))}</pre>
  `;
}

async function fetchText(objectId, path) {
  const params = new URLSearchParams({path});
  const data = await getJson(`/api/file_text/${encodeURIComponent(objectId)}?${params.toString()}`);
  return data;
}

function renderJson(detail) {
  const files = detail.json_files || [];
  $('jsonTab').innerHTML = `
    <div class="row">
      <select id="jsonSelect">${files.map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join('')}</select>
      <button id="loadJsonBtn">查看 JSON</button>
    </div>
    <pre id="jsonText">${files.length ? '选择文件后点击“查看 JSON”。' : '该样本没有 JSON 文件。'}</pre>
  `;
  const btn = $('loadJsonBtn');
  if (btn) btn.addEventListener('click', async () => {
    const path = $('jsonSelect').value;
    $('jsonText').textContent = '加载中...';
    try {
      const params = new URLSearchParams({path});
      const obj = await getJson(`/api/file_json/${encodeURIComponent(detail.summary.object_id)}?${params.toString()}`);
      $('jsonText').textContent = pretty(obj);
    } catch (e) {
      $('jsonText').textContent = `加载失败：${e.message}`;
    }
  });
}

function table(rows, cols) {
  if (!rows || !rows.length) return '<p class="muted">无数据</p>';
  return `<div class="table-wrap"><table><thead><tr>${cols.map(c => `<th>${esc(c.label)}</th>`).join('')}</tr></thead><tbody>` +
    rows.map(r => `<tr>${cols.map(c => `<td>${esc(r[c.key])}</td>`).join('')}</tr>`).join('') +
    '</tbody></table></div>';
}

function renderUrdf(detail) {
  const u = detail.urdf_summary;
  if (!u) {
    $('urdfTab').innerHTML = '<p class="muted">未找到 URDF 文件。</p>';
    return;
  }
  if (u.parse_error) {
    $('urdfTab').innerHTML = `<p>URDF 解析失败：${esc(u.parse_error)}</p><pre>${esc(u.raw_prefix)}</pre>`;
    return;
  }
  $('urdfTab').innerHTML = `
    <div class="grid two">
      ${statCard('URDF 文件', u.file)}
      ${statCard('robot name', u.robot_name || '-')}
      ${statCard('link 数量', u.link_count)}
      ${statCard('joint 数量', u.joint_count)}
    </div>
    <h3>Joints</h3>
    ${table(u.joints, [
      {key:'name', label:'name'}, {key:'type', label:'type'}, {key:'parent', label:'parent'}, {key:'child', label:'child'},
      {key:'origin_xyz', label:'origin xyz'}, {key:'origin_rpy', label:'origin rpy'}, {key:'axis_xyz', label:'axis'},
      {key:'limit_lower', label:'lower'}, {key:'limit_upper', label:'upper'},
    ])}
    <h3>Links / mesh refs</h3>
    ${table((u.links || []).map(x => ({name: x.name, mesh_refs: (x.mesh_refs || []).join(' | ')})), [
      {key:'name', label:'link name'}, {key:'mesh_refs', label:'mesh refs'}
    ])}
  `;
}

function renderFiles(detail) {
  const files = detail.files || [];
  $('filesTab').innerHTML = `
    <div class="file-list">
      ${files.map(f => `
        <div class="file-item">
          <span>${esc(f)}</span>
          <span>
            ${['.json','.urdf','.xml','.txt','.yaml','.yml','.obj','.mtl'].some(ext => f.toLowerCase().endsWith(ext)) ? `<button data-view="${esc(f)}">查看文本</button>` : ''}
            <a href="/api/file_raw/${encodeURIComponent(detail.summary.object_id)}?path=${encodeURIComponent(f)}" target="_blank">打开原文件</a>
          </span>
        </div>
      `).join('')}
    </div>
    <pre id="fileText" style="margin-top:12px; display:none;"></pre>
  `;
  document.querySelectorAll('[data-view]').forEach(btn => btn.addEventListener('click', async () => {
    const path = btn.getAttribute('data-view');
    const pre = $('fileText');
    pre.style.display = 'block';
    pre.textContent = '加载中...';
    try {
      const data = await fetchText(detail.summary.object_id, path);
      pre.textContent = `${data.path} (${data.size_bytes} bytes${data.truncated ? ', 已截断' : ''})\n\n${data.text}`;
    } catch (e) {
      pre.textContent = `加载失败：${e.message}`;
    }
  }));
}

async function ensureThree() {
  if (state.three) return state.three;
  const version = '0.160.0';
  try {
    const THREE = await import(`https://unpkg.com/three@${version}/build/three.module.js`);
    const { OrbitControls } = await import(`https://unpkg.com/three@${version}/examples/jsm/controls/OrbitControls.js`);
    const { OBJLoader } = await import(`https://unpkg.com/three@${version}/examples/jsm/loaders/OBJLoader.js`);
    state.three = { THREE, OrbitControls, OBJLoader };
    return state.three;
  } catch (e) {
    throw new Error('3D 预览需要浏览器能访问 unpkg.com 加载 three.js；当前只影响 3D 预览，JSON/URDF/文件查看不受影响。');
  }
}

function disposeViewer(container) {
  while (container.firstChild) container.removeChild(container.firstChild);
}

async function loadObjPreview(objectId, path) {
  const container = $('viewer3d');
  disposeViewer(container);
  container.textContent = '加载 3D 模型中...';
  const { THREE, OrbitControls, OBJLoader } = await ensureThree();
  disposeViewer(container);

  const width = container.clientWidth;
  const height = container.clientHeight;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.001, 1000);
  camera.position.set(1.6, 1.2, 1.6);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  container.appendChild(renderer.domElement);

  const light1 = new THREE.DirectionalLight(0xffffff, 1.0);
  light1.position.set(2, 3, 4);
  scene.add(light1);
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  scene.add(new THREE.GridHelper(2, 20));
  scene.add(new THREE.AxesHelper(0.5));

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const url = `/api/file_raw/${encodeURIComponent(objectId)}?path=${encodeURIComponent(path)}`;
  const loader = new OBJLoader();
  loader.load(url, (obj) => {
    obj.traverse(child => {
      if (child.isMesh) {
        child.material = new THREE.MeshStandardMaterial({ metalness: 0.0, roughness: 0.8 });
      }
    });
    scene.add(obj);
    const box = new THREE.Box3().setFromObject(obj);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    obj.position.sub(center);
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const scale = 1.5 / maxDim;
    obj.scale.setScalar(scale);
    camera.position.set(1.8, 1.4, 1.8);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
  }, undefined, (err) => {
    container.textContent = `OBJ 加载失败：${err.message || err}`;
  });

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}

function renderMesh(detail) {
  const objFiles = (detail.mesh_files || []).filter(f => f.toLowerCase().endsWith('.obj'));
  const images = detail.image_files || [];
  $('meshTab').innerHTML = `
    <div class="row">
      <select id="meshSelect">${objFiles.map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join('')}</select>
      <button id="loadMeshBtn">预览 OBJ</button>
      <span class="muted">当前仅内置 OBJ 预览；PLY/STL 可通过“文件树”打开原文件。</span>
    </div>
    <div id="viewer3d">${objFiles.length ? '选择 OBJ 后点击“预览 OBJ”。' : '该样本未发现 OBJ mesh。'}</div>
    <h3>图片文件</h3>
    <div class="file-list">
      ${images.map(f => `<div class="file-item"><span>${esc(f)}</span><a href="/api/file_raw/${encodeURIComponent(detail.summary.object_id)}?path=${encodeURIComponent(f)}" target="_blank">打开图片</a></div>`).join('') || '<p class="muted">未发现图片文件。</p>'}
    </div>
  `;
  const btn = $('loadMeshBtn');
  if (btn) btn.addEventListener('click', async () => {
    const f = $('meshSelect').value;
    try { await loadObjPreview(detail.summary.object_id, f); }
    catch (e) { $('viewer3d').textContent = e.message; }
  });
}

async function loadObject(objectId) {
  state.currentObjectId = objectId;
  document.querySelectorAll('.object-card').forEach(el => el.classList.toggle('active', el.dataset.id === objectId));
  $('emptyState').classList.add('hidden');
  $('detailContent').classList.remove('hidden');
  $('detailTitle').textContent = `Object ${objectId}`;
  $('detailSub').textContent = '加载中...';
  try {
    const detail = await getJson(`/api/object/${encodeURIComponent(objectId)}`);
    state.currentDetail = detail;
    $('detailTitle').textContent = `Object ${detail.summary.object_id}`;
    $('detailSub').textContent = `${detail.summary.category} · files=${detail.summary.file_count}`;
    renderSummary(detail);
    renderJson(detail);
    renderUrdf(detail);
    renderMesh(detail);
    renderFiles(detail);
  } catch (e) {
    $('detailSub').textContent = `加载失败：${e.message}`;
  }
}

function setupTabs() {
  document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-body').forEach(x => x.classList.remove('active'));
    btn.classList.add('active');
    $(btn.dataset.tab).classList.add('active');
  }));
}

function setupControls() {
  $('searchBtn').addEventListener('click', () => {
    state.q = $('searchInput').value.trim();
    state.category = $('categorySelect').value;
    state.offset = 0;
    loadObjects();
  });
  $('searchInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('searchBtn').click(); });
  $('categorySelect').addEventListener('change', () => $('searchBtn').click());
  $('prevBtn').addEventListener('click', () => { state.offset = Math.max(0, state.offset - state.limit); loadObjects(); });
  $('nextBtn').addEventListener('click', () => { state.offset += state.limit; loadObjects(); });
}

async function main() {
  setupTabs();
  setupControls();
  try {
    await loadStats();
    await loadObjects();
  } catch (e) {
    $('stats').innerHTML = `<pre>启动失败：${esc(e.message)}</pre>`;
  }
}

main();
