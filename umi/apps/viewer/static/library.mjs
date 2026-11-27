export async function api(url) {
  const response = await fetch(url, {cache: 'no-store'});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `读取失败（${response.status}）`);
  return body;
}
export function recordingLabel(name) {
  const match = name.match(/VID_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
  return match ? `${match[1]}.${match[2]}.${match[3]} · ${match[4]}:${match[5]}:${match[6]}` : name;
}

export function createLibrary({search, count, list, root, refreshButton, onSelect, onError}) {
  let catalog = [];
  let active = null;

  function render() {
    const filter = search.value.trim().toLowerCase();
    const entries = catalog.filter(entry => `${entry.name} ${entry.serial || ''} ${entry.path}`.toLowerCase().includes(filter));
    count.textContent = catalog.length;
    list.replaceChildren();
    if (!entries.length) {
      const empty = list.ownerDocument.createElement('p'); empty.className = 'empty-list';
      empty.textContent = filter ? '没有匹配的录像。' : '还没有可查看的轨迹。先运行 umi-scene run 处理任务，再刷新此列表。'; list.append(empty); return;
    }
    for (const [index, entry] of entries.entries()) {
      const button = list.ownerDocument.createElement('button'); button.className = `recording-item${active === entry.id ? ' active' : ''}`;
      button.setAttribute('aria-current', active === entry.id ? 'true' : 'false'); button.title = entry.reason || entry.path; button.disabled = entry.available === false;
      const top = list.ownerDocument.createElement('span'); top.className = 'record-index';
      const title = list.ownerDocument.createElement('span'); title.textContent = `TAKE ${String(index + 1).padStart(2, '0')}`;
      const status = list.ownerDocument.createElement('i'); status.className = `item-status${['failed', 'rejected_preflight', 'incomplete'].includes(entry.status) ? ' failed' : ''}`;
      top.append(title, status);
      const name = list.ownerDocument.createElement('strong'); name.textContent = recordingLabel(entry.name);
      const labels = {pending:'待处理', incomplete:'待补齐', failed:'处理失败', rejected_preflight:'原片检查未通过', ok:'已完成', skipped:'已完成', local_ok:'独立局部轨迹', local_skipped:'独立局部轨迹（复用）'};
      const sub = list.ownerDocument.createElement('small'); sub.textContent = `${entry.serial?.slice(-4) || '设备未知'}  /  ${Number.isFinite(entry.duration) ? entry.duration.toFixed(1) + ' s' : '时长未知'}  /  ${entry.rows ?? '—'} 帧 / ${labels[entry.status] || entry.status}`;
      if (entry.published && !entry.current) {
        const run = entry.published.run.split('/').pop();
        sub.textContent += ` · 显示上次发布 ${run}`;
        button.title = [entry.attempt?.reason, entry.published.path].filter(Boolean).join(' · ');
      }
      button.append(top, name, sub); button.addEventListener('click', () => onSelect(entry.id)); list.append(button);
    }
  }
  async function refresh(initial = false) {
    refreshButton.disabled = true;
    try {
      const result = await api('/api/recordings');
      catalog = result.recordings;
      root.textContent = result.root;
      root.title = result.root;
      render();
      if (initial && catalog.length) {
        const requested = new URL(location.href).searchParams.get('recording');
        const available = catalog.filter(item => item.available !== false);
        const entry = available.find(item => item.id === requested) || available[0];
        if (entry) await onSelect(entry.id);
      }
    } catch (error) { onError(error.message); }
    finally { refreshButton.disabled = false; }
  }

  refreshButton.addEventListener('click', () => refresh());
  search.addEventListener('input', render);
  return {refresh, setActive(id) { active = id; render(); }};
}
