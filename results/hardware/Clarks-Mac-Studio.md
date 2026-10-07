# Node: Clarks-Mac-Studio.local

| Field | Value |
|---|---|
| Chip | Apple M5 Max |
| CPU cores | 18 |
| GPU cores | 32 |
| RAM | 36.0 GiB (38654705664 bytes) |
| GPU max recommended working set | 28.1 GiB (30150672384 bytes) |
| MLX default memory limit | 34.2 GiB (36721970380 bytes) |
| macOS | 27.0.1 |
| Python | 3.12.15 |
| mlx | 0.32.3 |
| MLX default device | Device(gpu, 0) |
| Thunderbolt / bridge IPs | {'en1': '10.120.46.216'} |
| RDMA devices (ibv_devices) | device          	   node GUID
    ------          	----------------
    rdma_en2        	40bf9e42849fac05
    rdma_en9        	41bf9e42849fac05
    rdma_en3        	42bf9e42849fac05
    rdma_en4        	43bf9e42849fac05 |
| Collected | 2026-10-07T07:52:32-0500 |

## mx.metal.device_info()

```json
{
  "device_name": "Apple M5 Max",
  "max_recommended_working_set_size": 30150672384,
  "memory_size": 38654705664,
  "architecture": "applegpu_g17s",
  "max_buffer_length": 22613000192,
  "resource_limit": 499000
}
```

## Sanity check

```json
{
  "matmul_ok": true,
  "matmul_seconds": 0.0977,
  "matmul_result": 8589934592.0
}
```
