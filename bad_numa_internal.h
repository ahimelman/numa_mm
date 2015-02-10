#ifndef __SPARC_MM_NUMA_INTERNAL_H
#define __SPARC_MM_NUMA_INTERNAL_H

#include <linux/types.h>

#define NODE_MIN_SIZE (4*1024*1024)
#define NR_NODE_MEMBLKS (MAX_NUMNODES * 2)
#define FAKE_NODE_MIN_SIZE      ((unsigned long long)32 << 20)
#define FAKE_NODE_MIN_HASH_MASK (~(FAKE_NODE_MIN_SIZE - 1UL))
#define MAX_DMA32_PFN ((4UL * 1024 * 1024 * 1024) >> PAGE_SHIFT)
#define MAX_IO_APICS 128
#define MAX_LOCAL_APIC 32768

#define early_per_cpu_ptr(_name) (_name##_early_ptr)

#define early_per_cpu(_name, _cpu)                              \
        *(early_per_cpu_ptr(_name) ?                            \
                &early_per_cpu_ptr(_name)[_cpu] :               \
                &per_cpu(_name, _cpu))

#define DECLARE_EARLY_PER_CPU(_type, _name)                     \
        DECLARE_PER_CPU(_type, _name);                          \
        extern __typeof__(_type) *_name##_early_ptr;            \
        extern __typeof__(_type)  _name##_early_map[]

extern short __apicid_to_node[MAX_LOCAL_APIC];

struct numa_memblk {
	unsigned long long		start;
	unsigned long long		end;
	int			            nid;
};

struct numa_meminfo {
	int			nr_blks;
	struct numa_memblk	blk[NR_NODE_MEMBLKS];
};

void __init numa_remove_memblk_from(int idx, struct numa_meminfo *mi);
int __init numa_cleanup_meminfo(struct numa_meminfo *mi);
void __init numa_reset_distance(void);

void __init sparc_numa_init(void);

void __init numa_emulation(struct numa_meminfo *numa_meminfo,
			   int numa_dist_cnt);

#endif	/* __SPARC_MM_NUMA_INTERNAL_H */
