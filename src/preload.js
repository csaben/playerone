const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('po', {
  on: (channel, fn) => {
    if (['state', 'collapse', 'expand', 'toast'].includes(channel)) {
      ipcRenderer.on(channel, (_e, data) => fn(data));
    }
  },
  ignore: (v) => ipcRenderer.send('ui:ignore', v),
  size: (s) => ipcRenderer.send('ui:size', s),
  focus: () => ipcRenderer.send('ui:focus'),
  menu: () => ipcRenderer.send('ui:menu'),
  ready: () => ipcRenderer.send('ui:ready'),
  expanded: (v) => ipcRenderer.send('ui:expanded', v),
  attach: (key) => ipcRenderer.invoke('win:attach', key),
  detach: (key) => ipcRenderer.invoke('win:detach', key),
  record: (key) => ipcRenderer.invoke('rec:toggle', key),
  play: (key) => ipcRenderer.invoke('win:play', key),
  pickModel: (name) => ipcRenderer.invoke('model:pick', name),
  setPlay: (name, value) => ipcRenderer.invoke('play:set', name, value),
  devices: () => ipcRenderer.invoke('rec:devices'),
  setDevices: (mask) => ipcRenderer.invoke('rec:set-devices', mask),
  datasets: () => ipcRenderer.invoke('rec:datasets'),
  setDataset: (repo) => ipcRenderer.invoke('rec:set-dataset', repo),
  open: (dir) => ipcRenderer.invoke('rec:open', dir),
});
