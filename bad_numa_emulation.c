/*
 * NUMA emulation
 */
#include <linux/kernel.h>
#include <linux/errno.h>
#include <linux/topology.h>
#include <linux/memblock.h>
#include <linux/bootmem.h>
#include <asm/dma.h>

#include "numa_internal.h"

#include <linux/mm.h>
#include <linux/string.h>
#include <linux/init.h>
#include <linux/mmzone.h>
#include <linux/ctype.h>
#include <linux/module.h>
#include <linux/nodemask.h>
#include <linux/sched.h>

#include <linux/module.h>
#include <linux/hugetlb.h>
#include <linux/initrd.h>
#include <linux/swap.h>
#include <linux/pagemap.h>
#include <linux/poison.h>
#include <linux/fs.h>
#include <linux/seq_file.h>
#include <linux/kprobes.h>
#include <linux/cache.h>
#include <linux/sort.h>
#include <linux/ioport.h>
#include <linux/percpu.h>
#include <linux/gfp.h>

#include <asm/head.h>
#include <asm/page.h>
#include <asm/pgalloc.h>
#include <asm/pgtable.h>
#include <asm/oplib.h>
#include <asm/iommu.h>
#include <asm/io.h>
#include <asm/uaccess.h>
#include <asm/mmu_context.h>
#include <asm/tlbflush.h>
#include <asm/starfire.h>
#include <asm/tlb.h>
#include <asm/spitfire.h>
#include <asm/sections.h>
#include <asm/tsb.h>
#include <asm/hypervisor.h>
#include <asm/prom.h>
#include <asm/mdesc.h>
#include <asm/cpudata.h>
#include <asm/setup.h>
#include <asm/irq.h>

static int emu_nid_to_phys[MAX_NUMNODES];
static char *emu_cmdline __initdata;

static int numa_distance_cnt;
static unsigned char *numa_distance;
static struct numa_meminfo numa_meminfo __initdata;
nodemask_t numa_nodes_parsed __initdata;

/**
 * numa_remove_memblk_from - Remove one numa_memblk from a numa_meminfo
 * @idx: Index of memblk to remove
 * @mi: numa_meminfo to remove memblk from
 *
 * Remove @idx'th numa_memblk from @mi by shifting @mi->blk[] and
 * decrementing @mi->nr_blks.
 */
void __init numa_remove_memblk_from(int idx, struct numa_meminfo *mi)
{
    mi->nr_blks--;
    memmove(&mi->blk[idx], &mi->blk[idx + 1],
            (mi->nr_blks - idx) * sizeof(mi->blk[0]));
}

/**
 * numa_cleanup_meminfo - Cleanup a numa_meminfo
 * @mi: numa_meminfo to clean up
 *
 * Sanitize @mi by merging and removing unncessary memblks.  Also check for
 * conflicts and clear unused memblks.
 *
 * RETURNS:
 * 0 on success, -errno on failure.
 */
int __init numa_cleanup_meminfo(struct numa_meminfo *mi)
{
    const unsigned long long low = 0;
    const unsigned long long high = PFN_PHYS(max_pfn);
    int i, j, k;

    /* first, trim all entries */
    for (i = 0; i < mi->nr_blks; i++) {
        struct numa_memblk *bi = &mi->blk[i];

        /* make sure all blocks are inside the limits */
        bi->start = max(bi->start, low);
        bi->end = min(bi->end, high);

        /* and there's no empty block */
        if (bi->start >= bi->end)
            numa_remove_memblk_from(i--, mi);
    }

    /* merge neighboring / overlapping entries */
    for (i = 0; i < mi->nr_blks; i++) {
        struct numa_memblk *bi = &mi->blk[i];

        for (j = i + 1; j < mi->nr_blks; j++) {
            struct numa_memblk *bj = &mi->blk[j];
            unsigned long long start, end;

            /*
             * See whether there are overlapping blocks.  Whine
             * about but allow overlaps of the same nid.  They
             * will be merged below.
             */
            if (bi->end > bj->start && bi->start < bj->end) {
                if (bi->nid != bj->nid) {
                    pr_err("NUMA: node %d [mem %#010Lx-%#010Lx] overlaps with node %d [mem %#010Lx-%#010Lx]\n",
                            bi->nid, bi->start, bi->end - 1,
                            bj->nid, bj->start, bj->end - 1);
                    return -EINVAL;
                }
                pr_warning("NUMA: Warning: node %d [mem %#010Lx-%#010Lx] overlaps with itself [mem %#010Lx-%#010Lx]\n",
                        bi->nid, bi->start, bi->end - 1,
                        bj->start, bj->end - 1);
            }

            /*
             * Join together blocks on the same node, holes
             * between which don't overlap with memory on other
             * nodes.
             */
            if (bi->nid != bj->nid)
                continue;
            start = min(bi->start, bj->start);
            end = max(bi->end, bj->end);
            for (k = 0; k < mi->nr_blks; k++) {
                struct numa_memblk *bk = &mi->blk[k];

                if (bi->nid == bk->nid)
                    continue;
                if (start < bk->end && end > bk->start)
                    break;
            }
            if (k < mi->nr_blks)
                continue;
            printk(KERN_INFO "NUMA: Node %d [mem %#010Lx-%#010Lx] + [mem %#010Lx-%#010Lx] -> [mem %#010Lx-%#010Lx]\n",
                    bi->nid, bi->start, bi->end - 1, bj->start,
                    bj->end - 1, start, end - 1);
            bi->start = start;
            bi->end = end;
            numa_remove_memblk_from(j--, mi);
        }
    }

    /* clear unused ones */
    for (i = mi->nr_blks; i < ARRAY_SIZE(mi->blk); i++) {
        mi->blk[i].start = mi->blk[i].end = 0;
        mi->blk[i].nid = NUMA_NO_NODE;
    }

    return 0;
}

/**
 * numa_reset_distance - Reset NUMA distance table
 *
 * The current table is freed.  The next numa_set_distance() call will
 * create a new one.
 */
void __init numa_reset_distance(void)
{
    size_t size = numa_distance_cnt * numa_distance_cnt * sizeof(numa_distance[0]);

    /* numa_distance could be 1LU marking allocation failure, test cnt */
    if (numa_distance_cnt)
        memblock_free(__pa(numa_distance), size);
    numa_distance_cnt = 0;
    numa_distance = NULL;	/* enable table creation */
}

short __apicid_to_node[MAX_LOCAL_APIC] = {
    [0 ... MAX_LOCAL_APIC-1] = NUMA_NO_NODE
};

static inline void set_apicid_to_node(int apicid, short node)
{
    __apicid_to_node[apicid] = node;
}

/*
 * Set nodes, which have memory in @mi, in *@nodemask.
 */
static void __init numa_nodemask_from_meminfo(nodemask_t *nodemask,
        const struct numa_meminfo *mi)
{
    int i;

    for (i = 0; i < ARRAY_SIZE(mi->blk); i++)
        if (mi->blk[i].start != mi->blk[i].end &&
                mi->blk[i].nid != NUMA_NO_NODE)
            node_set(mi->blk[i].nid, *nodemask);
}

static void __init numa_clear_kernel_node_hotplug(void)
{
    int i, nid;
    nodemask_t numa_kernel_nodes = NODE_MASK_NONE;
    unsigned long start, end;
    struct memblock_region *r;

    /*
     * At this time, all memory regions reserved by memblock are
     * used by the kernel. Set the nid in memblock.reserved will
     * mark out all the nodes the kernel resides in.
     */
    for (i = 0; i < numa_meminfo.nr_blks; i++) {
        struct numa_memblk *mb = &numa_meminfo.blk[i];

        memblock_set_node(mb->start, mb->end - mb->start,
                &memblock.reserved, mb->nid);
    }

    /* Mark all kernel nodes. */
    for_each_memblock(reserved, r)
        node_set(r->nid, numa_kernel_nodes);

    /* Clear MEMBLOCK_HOTPLUG flag for memory in kernel nodes. */
    for (i = 0; i < numa_meminfo.nr_blks; i++) {
        nid = numa_meminfo.blk[i].nid;
        if (!node_isset(nid, numa_kernel_nodes))
            continue;

        start = numa_meminfo.blk[i].start;
        end = numa_meminfo.blk[i].end;

        memblock_clear_hotplug(start, end - start);
    }
}

/*
 * Sanity check to catch more bad NUMA configurations (they are amazingly
 * common).  Make sure the nodes cover all memory.
 */
static bool __init numa_meminfo_cover_memory(const struct numa_meminfo *mi)
{
    unsigned long long numaram, e820ram;
    int i;

    numaram = 0;
    for (i = 0; i < mi->nr_blks; i++) {
        unsigned long long s = mi->blk[i].start >> PAGE_SHIFT;
        unsigned long long e = mi->blk[i].end >> PAGE_SHIFT;
        numaram += e - s;
        numaram -= __absent_pages_in_range(mi->blk[i].nid, s, e);
        if ((s64)numaram < 0)
            numaram = 0;
    }

    e820ram = max_pfn - absent_pages_in_range(0, max_pfn);

    /* We seem to lose 3 pages somewhere. Allow 1M of slack. */
    if ((s64)(e820ram - numaram) >= (1 << (20 - PAGE_SHIFT))) {
        printk(KERN_ERR "NUMA: nodes only cover %LuMB of your %LuMB e820 RAM. Not used.\n",
                (numaram << PAGE_SHIFT) >> 20,
                (e820ram << PAGE_SHIFT) >> 20);
        return false;
    }
    return true;
}

/* Allocate NODE_DATA for a node on the local memory */
static void __init alloc_node_data(int nid)
{
    const size_t nd_size = roundup(sizeof(pg_data_t), PAGE_SIZE);
    unsigned long long nd_pa;
    void *nd;
    int tnid;

    /*
     * Allocate node data.  Try node-local memory and then any node.
     * Never allocate in DMA zone.
     */
    nd_pa = memblock_alloc_nid(nd_size, SMP_CACHE_BYTES, nid);
    if (!nd_pa) {
        nd_pa = __memblock_alloc_base(nd_size, SMP_CACHE_BYTES,
                MEMBLOCK_ALLOC_ACCESSIBLE);
        if (!nd_pa) {
            pr_err("Cannot find %zu bytes in node %d\n",
                    nd_size, nid);
            return;
        }
    }
    nd = __va(nd_pa);

    /* report and initialize */
    printk(KERN_INFO "NODE_DATA(%d) allocated [mem %#010Lx-%#010Lx]\n", nid,
            nd_pa, nd_pa + nd_size - 1);
    tnid = early_pfn_to_nid(nd_pa >> PAGE_SHIFT);
    if (tnid != nid)
        printk(KERN_INFO "    NODE_DATA(%d) on node %d\n", nid, tnid);

    node_data[nid] = nd;
    memset(NODE_DATA(nid), 0, sizeof(pg_data_t));

    node_set_online(nid);
}

static int __init numa_register_memblks(struct numa_meminfo *mi)
{
    unsigned long uninitialized_var(pfn_align);
    int i, nid;

    /* Account for nodes with cpus and no memory */
    node_possible_map = numa_nodes_parsed;
    numa_nodemask_from_meminfo(&node_possible_map, mi);
    if (WARN_ON(nodes_empty(node_possible_map)))
        return -EINVAL;

    for (i = 0; i < mi->nr_blks; i++) {
        struct numa_memblk *mb = &mi->blk[i];
        memblock_set_node(mb->start, mb->end - mb->start,
                &memblock.memory, mb->nid);
    }

    /*
     * At very early time, the kernel have to use some memory such as
     * loading the kernel image. We cannot prevent this anyway. So any
     * node the kernel resides in should be un-hotpluggable.
     *
     * And when we come here, alloc node data won't fail.
     */
    numa_clear_kernel_node_hotplug();

    /*
     * If sections array is gonna be used for pfn -> nid mapping, check
     * whether its granularity is fine enough.
     */
#ifdef NODE_NOT_IN_PAGE_FLAGS
    pfn_align = node_map_pfn_alignment();
    if (pfn_align && pfn_align < PAGES_PER_SECTION) {
        printk(KERN_WARNING "Node alignment %LuMB < min %LuMB, rejecting NUMA config\n",
                PFN_PHYS(pfn_align) >> 20,
                PFN_PHYS(PAGES_PER_SECTION) >> 20);
        return -EINVAL;
    }
#endif
    if (!numa_meminfo_cover_memory(mi))
        return -EINVAL;

    /* Finally register nodes. */
    for_each_node_mask(nid, node_possible_map) {
        unsigned long long start = PFN_PHYS(max_pfn);
        unsigned long long end = 0;

        for (i = 0; i < mi->nr_blks; i++) {
            if (nid != mi->blk[i].nid)
                continue;
            start = min(mi->blk[i].start, start);
            end = max(mi->blk[i].end, end);
        }

        if (start >= end)
            continue;

        /*
         * Don't confuse VM with a node that doesn't have the
         * minimum amount of memory:
         */
        if (end && (end - start) < NODE_MIN_SIZE)
            continue;

        alloc_node_data(nid);
    }

    /* Dump memblock with node info and return. */
    memblock_dump_all();
    return 0;
}

DECLARE_EARLY_PER_CPU(int, x86_cpu_to_node_map);
DEFINE_EARLY_PER_CPU(int, x86_cpu_to_node_map, NUMA_NO_NODE);
EXPORT_EARLY_PER_CPU_SYMBOL(x86_cpu_to_node_map);
DECLARE_PER_CPU(int, numa_node);

static inline int early_cpu_to_node(int cpu)
{
    return early_per_cpu(x86_cpu_to_node_map, cpu); 
}

void numa_set_node(int cpu, int node)
{
    int *cpu_to_node_map = early_per_cpu_ptr(x86_cpu_to_node_map);

    /* early setting, no percpu area yet */
    if (cpu_to_node_map) {
        cpu_to_node_map[cpu] = node;
        return;
    }

#ifdef CONFIG_DEBUG_PER_CPU_MAPS
    if (cpu >= nr_cpu_ids || !cpu_possible(cpu)) {
        printk(KERN_ERR "numa_set_node: invalid cpu# (%d)\n", cpu);
        dump_stack();
        return;
    }
#endif
    per_cpu(x86_cpu_to_node_map, cpu) = node;

    /*set_cpu_numa_node(cpu, node);*/
    per_cpu(numa_node, cpu) = node;
}

static void __init numa_init_array(void)
{
    int rr, i;

    rr = first_node(node_online_map);
    for (i = 0; i < nr_cpu_ids; i++) {
        if (early_cpu_to_node(i) != NUMA_NO_NODE)
            continue;
        numa_set_node(i, rr);
        rr = next_node(rr, node_online_map);
        if (rr == MAX_NUMNODES)
            rr = first_node(node_online_map);
    }
}


void numa_clear_node(int cpu)
{
    numa_set_node(cpu, NUMA_NO_NODE);
}

static int __init numa_init(int (*init_func)(void))
{
    int i;
    int ret;

    for (i = 0; i < MAX_LOCAL_APIC; i++)
        set_apicid_to_node(i, NUMA_NO_NODE);

    nodes_clear(numa_nodes_parsed);
    nodes_clear(node_possible_map);
    nodes_clear(node_online_map);
    memset(&numa_meminfo, 0, sizeof(numa_meminfo));
    WARN_ON(memblock_set_node(0, ULLONG_MAX, &memblock.memory,
                MAX_NUMNODES));
    WARN_ON(memblock_set_node(0, ULLONG_MAX, &memblock.reserved,
                MAX_NUMNODES));
    /* In case that parsing SRAT failed. */
    WARN_ON(memblock_clear_hotplug(0, ULLONG_MAX));
    numa_reset_distance();

    ret = init_func();
    if (ret < 0)
        return ret;

    /*
     * We reset memblock back to the top-down direction
     * here because if we configured ACPI_NUMA, we have
     * parsed SRAT in init_func(). It is ok to have the
     * reset here even if we did't configure ACPI_NUMA
     * or acpi numa init fails and fallbacks to dummy
     * numa init.
     */
    memblock_set_bottom_up(false);

    ret = numa_cleanup_meminfo(&numa_meminfo);
    if (ret < 0)
        return ret;

    numa_emulation(&numa_meminfo, numa_distance_cnt);

    ret = numa_register_memblks(&numa_meminfo);
    if (ret < 0)
        return ret;

    for (i = 0; i < nr_cpu_ids; i++) {
        int nid = early_cpu_to_node(i);

        if (nid == NUMA_NO_NODE)
            continue;
        if (!node_online(nid))
            numa_clear_node(i);
    }
    numa_init_array();

    return 0;
}

static int __init numa_add_memblk_to(int nid, unsigned long long start, unsigned long long end,
        struct numa_meminfo *mi)
{
    /* ignore zero length blks */
    if (start == end)
        return 0;

    /* whine about and ignore invalid blks */
    if (start > end || nid < 0 || nid >= MAX_NUMNODES) {
        pr_warning("NUMA: Warning: invalid memblk node %d [mem %#010Lx-%#010Lx]\n",
                nid, start, end - 1);
        return 0;
    }

    if (mi->nr_blks >= NR_NODE_MEMBLKS) {
        pr_err("NUMA: too many memblk ranges\n");
        return -EINVAL;
    }

    mi->blk[mi->nr_blks].start = start;
    mi->blk[mi->nr_blks].end = end;
    mi->blk[mi->nr_blks].nid = nid;
    mi->nr_blks++;
    return 0;
}

int __init numa_add_memblk(int nid, unsigned long long start, unsigned long long end)
{
    return numa_add_memblk_to(nid, start, end, &numa_meminfo);
}

/**
 * dummy_numa_init - Fallback dummy NUMA init
 *
 * Used if there's no underlying NUMA architecture, NUMA initialization
 * fails, or NUMA is disabled on the command line.
 *
 * Must online at least one node and add memory blocks that cover all
 * allowed memory.  This function must not fail.
 */
static int __init dummy_numa_init(void)
{
    printk(KERN_INFO "Faking a node at [mem %#018Lx-%#018Lx]\n",
            0LLU, PFN_PHYS(max_pfn) - 1);

    node_set(0, numa_nodes_parsed);
    numa_add_memblk(0, 0, PFN_PHYS(max_pfn));

    return 0;
}

/**
 * sparc_numa_init - Initialize NUMA
 *
 * Try each configured NUMA initialization method until one succeeds.  The
 * last fallback is dummy single node config encomapssing whole memory and
 * never fails.
 */
void __init sparc_numa_init(void)
{
    numa_init(dummy_numa_init);
}

void __init numa_emu_cmdline(char *str)
{
	emu_cmdline = str;
}

static int __init emu_find_memblk_by_nid(int nid, const struct numa_meminfo *mi)
{
	int i;

	for (i = 0; i < mi->nr_blks; i++)
		if (mi->blk[i].nid == nid)
			return i;
	return -ENOENT;
}

static unsigned long __init mem_hole_size(unsigned long long start, unsigned long long end)
{
	unsigned long start_pfn = PFN_UP(start);
	unsigned long end_pfn = PFN_DOWN(end);

	if (start_pfn < end_pfn)
		return PFN_PHYS(absent_pages_in_range(start_pfn, end_pfn));
	return 0;
}

/*
 * Sets up nid to range from @start to @end.  The return value is -errno if
 * something went wrong, 0 otherwise.
 */
static int __init emu_setup_memblk(struct numa_meminfo *ei,
				   struct numa_meminfo *pi,
				   int nid, int phys_blk, unsigned long long size)
{
	struct numa_memblk *eb = &ei->blk[ei->nr_blks];
	struct numa_memblk *pb = &pi->blk[phys_blk];

	if (ei->nr_blks >= NR_NODE_MEMBLKS) {
		pr_err("NUMA: Too many emulated memblks, failing emulation\n");
		return -EINVAL;
	}

	ei->nr_blks++;
	eb->start = pb->start;
	eb->end = pb->start + size;
	eb->nid = nid;

	if (emu_nid_to_phys[nid] == NUMA_NO_NODE)
		emu_nid_to_phys[nid] = nid;

	pb->start += size;
	if (pb->start >= pb->end) {
		WARN_ON_ONCE(pb->start > pb->end);
		numa_remove_memblk_from(phys_blk, pi);
	}

	printk(KERN_INFO "Faking node %d at [mem %#018Lx-%#018Lx] (%LuMB)\n",
	       nid, eb->start, eb->end - 1, (eb->end - eb->start) >> 20);
	return 0;
}

/*
 * Sets up nr_nodes fake nodes interleaved over physical nodes ranging from addr
 * to max_addr.  The return value is the number of nodes allocated.
 */
static int __init split_nodes_interleave(struct numa_meminfo *ei,
					 struct numa_meminfo *pi,
					 unsigned long long addr, unsigned long long max_addr, int nr_nodes)
{
	nodemask_t physnode_mask = NODE_MASK_NONE;
	unsigned long long size;
	int big;
	int nid = 0;
	int i, ret;

	if (nr_nodes <= 0)
		return -1;
	if (nr_nodes > MAX_NUMNODES) {
		pr_info("numa=fake=%d too large, reducing to %d\n",
			nr_nodes, MAX_NUMNODES);
		nr_nodes = MAX_NUMNODES;
	}

	/*
	 * Calculate target node size.  x86_32 freaks on __udivdi3() so do
	 * the division in ulong number of pages and convert back.
	 */
	size = max_addr - addr - mem_hole_size(addr, max_addr);
	size = PFN_PHYS((unsigned long)(size >> PAGE_SHIFT) / nr_nodes);

	/*
	 * Calculate the number of big nodes that can be allocated as a result
	 * of consolidating the remainder.
	 */
	big = ((size & ~FAKE_NODE_MIN_HASH_MASK) * nr_nodes) /
		FAKE_NODE_MIN_SIZE;

	size &= FAKE_NODE_MIN_HASH_MASK;
	if (!size) {
		pr_err("Not enough memory for each node.  "
			"NUMA emulation disabled.\n");
		return -1;
	}

	for (i = 0; i < pi->nr_blks; i++)
		node_set(pi->blk[i].nid, physnode_mask);

	/*
	 * Continue to fill physical nodes with fake nodes until there is no
	 * memory left on any of them.
	 */
	while (nodes_weight(physnode_mask)) {
		for_each_node_mask(i, physnode_mask) {
			unsigned long long dma32_end = PFN_PHYS(MAX_DMA32_PFN);
			unsigned long long start, limit, end;
			int phys_blk;

			phys_blk = emu_find_memblk_by_nid(i, pi);
			if (phys_blk < 0) {
				node_clear(i, physnode_mask);
				continue;
			}
			start = pi->blk[phys_blk].start;
			limit = pi->blk[phys_blk].end;
			end = start + size;

			if (nid < big)
				end += FAKE_NODE_MIN_SIZE;

			/*
			 * Continue to add memory to this fake node if its
			 * non-reserved memory is less than the per-node size.
			 */
			while (end - start - mem_hole_size(start, end) < size) {
				end += FAKE_NODE_MIN_SIZE;
				if (end > limit) {
					end = limit;
					break;
				}
			}

			/*
			 * If there won't be at least FAKE_NODE_MIN_SIZE of
			 * non-reserved memory in ZONE_DMA32 for the next node,
			 * this one must extend to the boundary.
			 */
			if (end < dma32_end && dma32_end - end -
			    mem_hole_size(end, dma32_end) < FAKE_NODE_MIN_SIZE)
				end = dma32_end;

			/*
			 * If there won't be enough non-reserved memory for the
			 * next node, this one must extend to the end of the
			 * physical node.
			 */
			if (limit - end - mem_hole_size(end, limit) < size)
				end = limit;

			ret = emu_setup_memblk(ei, pi, nid++ % nr_nodes,
					       phys_blk,
					       min(end, limit) - start);
			if (ret < 0)
				return ret;
		}
	}
	return 0;
}

/*
 * Returns the end address of a node so that there is at least `size' amount of
 * non-reserved memory or `max_addr' is reached.
 */
static unsigned long long __init find_end_of_node(unsigned long long start, unsigned long long max_addr, unsigned long long size)
{
	unsigned long long end = start + size;

	while (end - start - mem_hole_size(start, end) < size) {
		end += FAKE_NODE_MIN_SIZE;
		if (end > max_addr) {
			end = max_addr;
			break;
		}
	}
	return end;
}

/*
 * Sets up fake nodes of `size' interleaved over physical nodes ranging from
 * `addr' to `max_addr'.  The return value is the number of nodes allocated.
 */
static int __init split_nodes_size_interleave(struct numa_meminfo *ei,
					      struct numa_meminfo *pi,
					      unsigned long long addr, unsigned long long max_addr, unsigned long long size)
{
	nodemask_t physnode_mask = NODE_MASK_NONE;
	unsigned long long min_size;
	int nid = 0;
	int i, ret;

	if (!size)
		return -1;
	/*
	 * The limit on emulated nodes is MAX_NUMNODES, so the size per node is
	 * increased accordingly if the requested size is too small.  This
	 * creates a uniform distribution of node sizes across the entire
	 * machine (but not necessarily over physical nodes).
	 */
	min_size = (max_addr - addr - mem_hole_size(addr, max_addr)) / MAX_NUMNODES;
	min_size = max(min_size, FAKE_NODE_MIN_SIZE);
	if ((min_size & FAKE_NODE_MIN_HASH_MASK) < min_size)
		min_size = (min_size + FAKE_NODE_MIN_SIZE) &
						FAKE_NODE_MIN_HASH_MASK;
	if (size < min_size) {
		pr_err("Fake node size %LuMB too small, increasing to %LuMB\n",
			size >> 20, min_size >> 20);
		size = min_size;
	}
	size &= FAKE_NODE_MIN_HASH_MASK;

	for (i = 0; i < pi->nr_blks; i++)
		node_set(pi->blk[i].nid, physnode_mask);

	/*
	 * Fill physical nodes with fake nodes of size until there is no memory
	 * left on any of them.
	 */
	while (nodes_weight(physnode_mask)) {
		for_each_node_mask(i, physnode_mask) {
			unsigned long long dma32_end = PFN_PHYS(MAX_DMA32_PFN);
			unsigned long long start, limit, end;
			int phys_blk;

			phys_blk = emu_find_memblk_by_nid(i, pi);
			if (phys_blk < 0) {
				node_clear(i, physnode_mask);
				continue;
			}
			start = pi->blk[phys_blk].start;
			limit = pi->blk[phys_blk].end;

			end = find_end_of_node(start, limit, size);
			/*
			 * If there won't be at least FAKE_NODE_MIN_SIZE of
			 * non-reserved memory in ZONE_DMA32 for the next node,
			 * this one must extend to the boundary.
			 */
			if (end < dma32_end && dma32_end - end -
			    mem_hole_size(end, dma32_end) < FAKE_NODE_MIN_SIZE)
				end = dma32_end;

			/*
			 * If there won't be enough non-reserved memory for the
			 * next node, this one must extend to the end of the
			 * physical node.
			 */
			if (limit - end - mem_hole_size(end, limit) < size)
				end = limit;

			ret = emu_setup_memblk(ei, pi, nid++ % MAX_NUMNODES,
					       phys_blk,
					       min(end, limit) - start);
			if (ret < 0)
				return ret;
		}
	}
	return 0;
}

static int __init numa_alloc_distance(void)
{
    nodemask_t nodes_parsed;
    size_t size;
    int i, j, cnt = 0;
    unsigned long long phys;

    /* size the new table and allocate it */
    nodes_parsed = numa_nodes_parsed;
    numa_nodemask_from_meminfo(&nodes_parsed, &numa_meminfo);

    for_each_node_mask(i, nodes_parsed)
        cnt = i;
    cnt++;
    size = cnt * cnt * sizeof(numa_distance[0]);

    phys = memblock_find_in_range(0, PFN_PHYS(max_pfn_mapped),
            size, PAGE_SIZE);
    if (!phys) {
        pr_warning("NUMA: Warning: can't allocate distance table!\n");
        /* don't retry until explicitly reset */
        numa_distance = (void *)1LU;
        return -ENOMEM;
    }
    memblock_reserve(phys, size);

    numa_distance = __va(phys);
    numa_distance_cnt = cnt;

    /* fill with the default distances */
    for (i = 0; i < cnt; i++)
        for (j = 0; j < cnt; j++)
            numa_distance[i * cnt + j] = i == j ?
                LOCAL_DISTANCE : REMOTE_DISTANCE;
    printk(KERN_DEBUG "NUMA: Initialized distance table, cnt=%d\n", cnt);

    return 0;
}

void __init numa_set_distance(int from, int to, int distance)
{
    if (!numa_distance && numa_alloc_distance() < 0)
        return;

    if (from >= numa_distance_cnt || to >= numa_distance_cnt ||
            from < 0 || to < 0) {
        pr_warn_once("NUMA: Warning: node ids are out of bound, from=%d to=%d distance=%d\n",
                from, to, distance);
        return;
    }

    if ((unsigned short)distance != distance ||
            (from == to && distance != LOCAL_DISTANCE)) {
        pr_warn_once("NUMA: Warning: invalid distance parameter, from=%d to=%d distance=%d\n",
                from, to, distance);
        return;
    }

    numa_distance[from * numa_distance_cnt + to] = distance;
}

/**
 * numa_emulation - Emulate NUMA nodes
 * @numa_meminfo: NUMA configuration to massage
 * @numa_dist_cnt: The size of the physical NUMA distance table
 *
 * Emulate NUMA nodes according to the numa=fake kernel parameter.
 * @numa_meminfo contains the physical memory configuration and is modified
 * to reflect the emulated configuration on success.  @numa_dist_cnt is
 * used to determine the size of the physical distance table.
 *
 * On success, the following modifications are made.
 *
 * - @numa_meminfo is updated to reflect the emulated nodes.
 *
 * - __apicid_to_node[] is updated such that APIC IDs are mapped to the
 *   emulated nodes.
 *
 * - NUMA distance table is rebuilt to represent distances between emulated
 *   nodes.  The distances are determined considering how emulated nodes
 *   are mapped to physical nodes and match the actual distances.
 *
 * - emu_nid_to_phys[] reflects how emulated nodes are mapped to physical
 *   nodes.  This is used by numa_add_cpu() and numa_remove_cpu().
 *
 * If emulation is not enabled or fails, emu_nid_to_phys[] is filled with
 * identity mapping and no other modification is made.
 */
void __init numa_emulation(struct numa_meminfo *numa_meminfo, int numa_dist_cnt)
{
	static struct numa_meminfo ei __initdata;
	static struct numa_meminfo pi __initdata;
	const unsigned long long max_addr = PFN_PHYS(max_pfn);
	unsigned char *phys_dist = NULL;
	size_t phys_size = numa_dist_cnt * numa_dist_cnt * sizeof(phys_dist[0]);
	int max_emu_nid, dfl_phys_nid;
	int i, j, ret;

	if (!emu_cmdline)
		goto no_emu;

	memset(&ei, 0, sizeof(ei));
	pi = *numa_meminfo;

	for (i = 0; i < MAX_NUMNODES; i++)
		emu_nid_to_phys[i] = NUMA_NO_NODE;

	/*
	 * If the numa=fake command-line contains a 'M' or 'G', it represents
	 * the fixed node size.  Otherwise, if it is just a single number N,
	 * split the system RAM into N fake nodes.
	 */
	if (strchr(emu_cmdline, 'M') || strchr(emu_cmdline, 'G')) {
		unsigned long long size;

		size = memparse(emu_cmdline, &emu_cmdline);
		ret = split_nodes_size_interleave(&ei, &pi, 0, max_addr, size);
	} else {
		unsigned long n;

		n = simple_strtoul(emu_cmdline, &emu_cmdline, 0);
		ret = split_nodes_interleave(&ei, &pi, 0, max_addr, n);
	}
	if (*emu_cmdline == ':')
		emu_cmdline++;

	if (ret < 0)
		goto no_emu;

	if (numa_cleanup_meminfo(&ei) < 0) {
		pr_warning("NUMA: Warning: constructed meminfo invalid, disabling emulation\n");
		goto no_emu;
	}

	/* copy the physical distance table */
	if (numa_dist_cnt) {
		unsigned long long phys;

		phys = memblock_find_in_range(0, PFN_PHYS(max_pfn_mapped),
					      phys_size, PAGE_SIZE);
		if (!phys) {
			pr_warning("NUMA: Warning: can't allocate copy of distance table, disabling emulation\n");
			goto no_emu;
		}
		memblock_reserve(phys, phys_size);
		phys_dist = __va(phys);

		for (i = 0; i < numa_dist_cnt; i++)
			for (j = 0; j < numa_dist_cnt; j++)
				phys_dist[i * numa_dist_cnt + j] =
					node_distance(i, j);
	}

	/*
	 * Determine the max emulated nid and the default phys nid to use
	 * for unmapped nodes.
	 */
	max_emu_nid = 0;
	dfl_phys_nid = NUMA_NO_NODE;
	for (i = 0; i < ARRAY_SIZE(emu_nid_to_phys); i++) {
		if (emu_nid_to_phys[i] != NUMA_NO_NODE) {
			max_emu_nid = i;
			if (dfl_phys_nid == NUMA_NO_NODE)
				dfl_phys_nid = emu_nid_to_phys[i];
		}
	}
	if (dfl_phys_nid == NUMA_NO_NODE) {
		pr_warning("NUMA: Warning: can't determine default physical node, disabling emulation\n");
		goto no_emu;
	}

	/* commit */
	*numa_meminfo = ei;

	/*
	 * Transform __apicid_to_node table to use emulated nids by
	 * reverse-mapping phys_nid.  The maps should always exist but fall
	 * back to zero just in case.
	 */
	for (i = 0; i < ARRAY_SIZE(__apicid_to_node); i++) {
		if (__apicid_to_node[i] == NUMA_NO_NODE)
			continue;
		for (j = 0; j < ARRAY_SIZE(emu_nid_to_phys); j++)
			if (__apicid_to_node[i] == emu_nid_to_phys[j])
				break;
		__apicid_to_node[i] = j < ARRAY_SIZE(emu_nid_to_phys) ? j : 0;
	}

	/* make sure all emulated nodes are mapped to a physical node */
	for (i = 0; i < ARRAY_SIZE(emu_nid_to_phys); i++)
		if (emu_nid_to_phys[i] == NUMA_NO_NODE)
			emu_nid_to_phys[i] = dfl_phys_nid;

	/* transform distance table */
	numa_reset_distance();
	for (i = 0; i < max_emu_nid + 1; i++) {
		for (j = 0; j < max_emu_nid + 1; j++) {
			int physi = emu_nid_to_phys[i];
			int physj = emu_nid_to_phys[j];
			int dist;

			if (get_option(&emu_cmdline, &dist) == 2)
				;
			else if (physi >= numa_dist_cnt || physj >= numa_dist_cnt)
				dist = physi == physj ?
					LOCAL_DISTANCE : REMOTE_DISTANCE;
			else
				dist = phys_dist[physi * numa_dist_cnt + physj];

			numa_set_distance(i, j, dist);
		}
	}

	/* free the copied physical distance table */
	if (phys_dist)
		memblock_free(__pa(phys_dist), phys_size);
	return;

no_emu:
	/* No emulation.  Build identity emu_nid_to_phys[] for numa_add_cpu() */
	for (i = 0; i < ARRAY_SIZE(emu_nid_to_phys); i++)
		emu_nid_to_phys[i] = i;
}

#ifndef CONFIG_DEBUG_PER_CPU_MAPS
void numa_add_cpu(int cpu)
{
	int physnid, nid;

	nid = early_cpu_to_node(cpu);
	BUG_ON(nid == NUMA_NO_NODE || !node_online(nid));

	physnid = emu_nid_to_phys[nid];

	/*
	 * Map the cpu to each emulated node that is allocated on the physical
	 * node of the cpu's apic id.
	 */
	for_each_online_node(nid)
		if (emu_nid_to_phys[nid] == physnid)
			cpumask_set_cpu(cpu, node_to_cpumask_map[nid]);
}

void numa_remove_cpu(int cpu)
{
	int i;

	for_each_online_node(i)
		cpumask_clear_cpu(cpu, node_to_cpumask_map[i]);
}
#else	/* !CONFIG_DEBUG_PER_CPU_MAPS */
static void numa_set_cpumask(int cpu, bool enable)
{
	int nid, physnid;

	nid = early_cpu_to_node(cpu);
	if (nid == NUMA_NO_NODE) {
		/* early_cpu_to_node() already emits a warning and trace */
		return;
	}

	physnid = emu_nid_to_phys[nid];

	for_each_online_node(nid) {
		if (emu_nid_to_phys[nid] != physnid)
			continue;

		debug_cpumask_set_cpu(cpu, nid, enable);
	}
}

void numa_add_cpu(int cpu)
{
	numa_set_cpumask(cpu, true);
}

void numa_remove_cpu(int cpu)
{
	numa_set_cpumask(cpu, false);
}
#endif	/* !CONFIG_DEBUG_PER_CPU_MAPS */
