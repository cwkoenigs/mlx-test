# Node: Clarks-Mac-Studio.local

| Field | Value |
|---|---|
| Chip | Apple M5 Max |
| CPU cores | 18 |
| GPU cores | 40 |
| RAM | 48.0 GiB (51539607552 bytes) |
| GPU max recommended working set | 37.4 GiB (40200896512 bytes) |
| MLX default memory limit | 45.6 GiB (48962627174 bytes) |
| macOS | 27.0.1 |
| Python | 3.12.15 |
| mlx | 0.32.3 |
| MLX default device | Device(gpu, 0) |
| Thunderbolt / bridge IPs | {'bridge0': '169.254.84.32', 'en1': '10.120.49.70', 'en5': '169.254.47.144'} |
| RDMA devices (ibv_devices) | device          	   node GUID
    ------          	----------------
    rdma_en2        	0042c4dff605ac05
    rdma_en9        	0142c4dff605ac05
    rdma_en3        	0242c4dff605ac05
    rdma_en4        	0342c4dff605ac05 |
| Collected | 2026-10-08T08:04:28-0500 |

## mx.device_info()

```json
{
  "device_name": "Apple M5 Max",
  "max_recommended_working_set_size": 40200896512,
  "memory_size": 51539607552,
  "architecture": "applegpu_g17s",
  "max_buffer_length": 30150672384,
  "resource_limit": 499000
}
```

## Sanity check

```json
{
  "matmul_ok": true,
  "matmul_seconds": 0.0074,
  "matmul_result": 8589934592.0
}
```
