import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer():
	"""
	Replay buffer for training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda:0')
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		num_new_eps = len(td)
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def _prepare_batch(self, td, info):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "task", "expert_value", "expert_action_dist", \
      			"episode", strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		task = td.get('task', None)
		expert_value = td.get('expert_value')[1:].unsqueeze(-1).contiguous()
		expert_action_dist = td.get('expert_action_dist')[1:].contiguous()
		episode = td.get('episode')[1:].contiguous()
		if task is not None:
			task = task[0].contiguous()
		info = TensorDict({
			"index": info['index'][0].view(-1, self.cfg.horizon+1).permute(1,0),
			"expert_value": expert_value,
			"expert_action_dist": expert_action_dist,
			"episode": episode
		})
		return obs, action, reward, task, info

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		data, info = self._buffer.sample(return_info=True)
		td = data.view(-1, self.cfg.horizon+1).permute(1, 0)
   
		# sanity check for memory leak of buffer (just for debugging)
		nan_idx = torch.isnan(data['expert_action_dist'])
		if nan_idx.sum().item() % self.cfg.action_dim != 0: 
			print("!!!unexpected nan numbers when sample():", nan_idx.sum().item()/self.cfg.action_dim, flush=True)
		
		return self._prepare_batch(td, info)
