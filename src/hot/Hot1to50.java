package hot;

import java.util.*;

public class Hot1to50 {
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public static class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    //注意：数组、栈里存的/要存的是下标还是具体的数

    /**
     * 哈希
     * 1.两数之和：秒,注意用哈希的前提是只有一个解，与三数之和区分开
     * 2.字母异位词分组：没给哈希提示肯定想不到，还得练
     * 3.最长连续序列：set，没写明白,24年九月，没想出来
     * 双指针
     * 4.移动零:秒
     * 5.盛最多水的容器：移动短板，ok
     * 6.三数之和：注意去重
     * 7.接雨水：开两个数组存左边最高和右边最低，秒了，优化成双指针费劲（大小关系确定算哪个位置的水），单调栈更费劲（相当于横着算）
     * 滑动窗口
     * 8.无重复字符的最长子串，set想的还挺明白，hash也还行但有个易错点
     * 9.找到字符串中所有字母异位词: 定长滑动窗口，优化点在于如何判断是异位词，不是先排序然后比较char[]相等了，而是计算每个char出现的个数(对比76)
     * 字串
     * 10.和为k的子数组：前缀和+哈希表，写的不够简洁
     * 11.滑动窗口最大值：双端队列，刚做过但是还是没做出来
     * 12.最小覆盖子串：滑动窗口 hard，9的升级版，固定窗口变可变窗口
     * 普通数组
     * 13.最大子数组和：动态规划秒，还有个分治的解法没想到（效率更差）（求一个数组最大子列和，可以把他从mid分开，然后最大要么左边取，要么右边，要么跨越，跨越就是左边包含右的最大+右边加包含左的最大）
     * 14.合并区间：注意几个api
     * 15.轮转数组：要求三种解法，额外数组 +  +o（1）空间:三次翻转
     * 16.除自身以外数组的乘积:有点前缀和的思想
     * 17.缺失的第一个正数：原地哈希(自定义哈希)，好思想
     * 矩阵
     * 18.矩阵置零，理清优化思路，知道为什么这么想
     * 19.螺旋矩阵：模拟
     * 20.旋转图像：模拟
     * 21.搜索二维矩阵2：还是没掌握好这个二分思想，找两个方向，一个方向变大，一个方向变小
     */

    // 3 第一反应排序，但题目要求线性时间复杂度，排序的目的在于快速判断当前数的下一个数是否存在，否则需要遍历，有没有办法能直接判断呢，那就是哈希
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        // 遍历nums，以当前num为结尾，一直往前找，最多能找到几个
        // 为了减少重复搜索，若num+1在set，就不找了
        int res = 0;//空集返回0
        for (int num : nums) {
            if (set.contains(num + 1)) {
                continue;
            }
            int count = 1;
            while (set.contains(num - 1)) {
                count++;
                num--;
            }
            res = Math.max(count, res);
        }
        return res;
    }

    static class double_pointer {
        // 4.移动零，指针指示器指向最后一个非零元素的下标
        // 补充：没必要swap，把所有非0元素挪到前面，最后剩下置零即可
        // 盛水最多的容器，三段式最简洁，因为用哪个计算面积就要移动哪条边
        // 三数之和：去重用for内continue比内部再写一个while好，因为while还需要额外判断是否越界，而continue直接用for的就行
        //接雨水: 2024-1010 只想出了两个数组，双指针优化没想明白
        public int trap(int[] height) {
            int n = height.length, left = 1, right = n - 2;
            int sum = 0;
            // 需要两个额外变量，而非直接比较 right和 right-1这种
            int leftMax = height[0], rightMax = height[n - 1];
            while (left <= right) {
                if (leftMax < rightMax) {
                    // left的左边最大一定是leftMax，但右边最大不一定是当前的rightMax，因为在right向左移的这个过程，他可能还会变大
                    // 而当前位置竖着能接多少水取决于leftMax和rightMax中较小的那个
                    sum += Math.max(leftMax - height[left], 0);
                    leftMax = Math.max(height[left], leftMax);
                    left++;
                } else {
                    sum += Math.max(rightMax - height[right], 0);
                    rightMax = Math.max(rightMax, height[right]);
                    right--;
                }
            }
            return sum;
        }

        public int trap2(int[] height) {
            Stack<Integer> stack = new Stack<>();
            stack.push(0);
            int sum = 0;
            // 严格单调递减栈，栈顶元素为底，底出栈，新的栈顶作为一个边，i作为另一个边
            for (int i = 1; i < height.length; i++) {
                // 如果大于栈顶元素,栈顶出栈作为底
                while (height[i] >= height[stack.peek()]) { //到底是大于还是大于等于,貌似都可以
                    int bottom = stack.pop();
                    if (stack.isEmpty()) {
                        break; //不能if(!empty)，会空栈异常
                    }
                    int h = Math.min(height[i], height[stack.peek()]) - height[bottom];
                    sum += h * (i - stack.peek() - 1);

                }
                stack.push(i);
            }
            return sum;
        }
    }

    static class sliding_window {
        // 8无重复字符的最长字串
        public int lengthOfLongestSubstring(String s) {
            // 注意 ""和 " "
            if (s.isEmpty()) return 0;
            Map<Character, Integer> map = new HashMap<>();
            int res = 0;
            int left = 0;
            // i 指向无重复字符的右端点，left指向左端点, 一定不能写成 s.toCharArray.length(),效率极差
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (map.containsKey(c)) {
                    // 如果这个char没有被之前的滑动丢弃，left就用定位+1，否则仍用当前left
                    left = Math.max(map.get(c) + 1, left);
                }
                map.put(c, i);
                res = Math.max(res, i - left + 1);
            }
            return res;
        }

        public List<Integer> findAnagrams(String s, String p) {
            // 如果选择转数组再排序再转String再比较(即使可以直接比数组)，太慢了，光排序要 s*plogp
            // 如果用数组
            List<Integer> res = new ArrayList<>();
            if (s.length() < p.length()) return res;
            char[] need = new char[26];
            char[] have = new char[26];
            for (int i = 0; i < p.length(); i++) {
                need[p.charAt(i) - 'a']++;
                have[s.charAt(i) - 'a']++;
            }
            if (Arrays.equals(need, have)) {
                res.add(0);
            }
            for (int i = 0; i < s.length() - p.length(); i++) {
                have[s.charAt(i) - 'a']--;
                have[s.charAt(i + p.length()) - 'a']++;
                if (Arrays.equals(need, have)) {
                    res.add(i + 1);
                }
            }
            return res;
        }
    }

    static class subStirng {
        // 2024-10-18思路：因为要求连续，所以还是一个滑动窗口。需要保证子组的和随着窗口的增大而增大，即单调变化
        // 如果全是正数才能用滑动窗口，和大了就左面缩一下；但本题可能有负数。
        // 滑动窗口也写的吭哧憋度，所以留了个版本在这
        // 又把前缀和忘了
        public int subarraySum_x(int[] nums, int k) {
            int left = 0, right = 0;
            int curSum = 0;
            int res = 0;
            while (right < nums.length) {
                curSum += nums[right];
                while (left <= right && curSum >= k) {
                    if (curSum == k) {
                        res++;
                    }
                    curSum -= nums[left];
                    left++;
                }
                right++;
            }
            return res;
        }

        public int subarraySum(int[] nums, int k) {
            //suffix 后缀
            // 数组可以优化掉
            int[] prefixSum = new int[nums.length];
            // 以当前元素为结尾，包含该元素的前缀和
            prefixSum[0] = nums[0];
            for (int i = 1; i < nums.length; i++) {
                prefixSum[i] += prefixSum[i - 1] + nums[i];
            }
            Map<Integer, Integer> map = new HashMap<>();
            // 什么都不选，有一种
            map.put(0, 1);
            int res = 0;
            for (int i = 0; i < nums.length; i++) {
                if (map.containsKey(prefixSum[i] - k)) {
                    res += map.get(prefixSum[i] - k);
                }
                map.put(prefixSum[i], map.getOrDefault(prefixSum[i], 0) + 1);
            }
            return res;
        }

        public int subarraySum_20240417(int[] nums, int k) {
            //1.暴力，找所有子数组o(n^3)(子数组n^2，双层for,遍历子数组求和o(n)) 所以想到
            // 遍历左边界的同时求和，这样就降了一维，想到该思路类似前缀和，前缀和也可以降一维
            // 前缀和加哈希表o(n)，哈希表记录所需前缀和是否出现过及出现次数
            //感觉双指针滑动窗口可以，试一下，可能有负数，没法滑动窗口，同时滑动窗口会漏0
            int preSum = 0;
            int res = 0;
            Map<Integer, Integer> map = new HashMap<>();
            map.put(0, 1);//别忘，前缀和为0的出现次数为1
            for (int i = 0; i < nums.length; i++) {
                preSum += nums[i];
                if (map.containsKey(preSum - k)) res += map.get(preSum - k);//先if，因为k可能得0
                map.put(preSum, map.getOrDefault(preSum, 0) + 1);
            }
            return res;
        }

        // 滑动窗口最大值
        public int[] maxSlidingWindow(int[] nums, int k) {
            Deque<Integer> deque = new LinkedList<>();
            int[] res = new int[nums.length - k + 1];
            // 把下标放进去，队底为最大的元素
            for (int i = 0; i < nums.length; i++) {
                while (!deque.isEmpty() && nums[i] > nums[deque.peekLast()]) {
                    deque.pollLast();
                }
                // 注意不是push
                deque.addLast(i);
                if (deque.peekFirst() == i - k) {
                    deque.pollFirst();
                }
                if (i >= k - 1) res[i - k + 1] = nums[deque.peekFirst()];
            }
            return res;
            //法2：利用api创建一个优先级队列，然后每次检验栈顶元素是否在窗口内，最后取出栈顶元素
            //PriorityQueue<int[]> pq = new PriorityQueue<>((a,b) -> b[0]-a[0]); pq.add(new int[]{nums[i],i});
        }

        // 最小覆盖字串,想到用数组去表征t出现的字符，然后滑动窗口，向右扩展直至满足需求，然后左收缩，直至不满足需求
        // 但不行，因为还有大写字母和数字，字符相减计算的是unicode，但ascll码范围内，unicode与ascll相同，
        // a-z是97 - 122，A到Z是65-90，如果要用数组要int[128],然后 int[c-'A']
        // 也可直接need[t.charAt(i)]++;这个时候char代表的是ascll码
        // 没想到用map去表征，不过即使用两个map还是很复杂，left和right每移动一下，就需要遍历两个map去比
        // 现在慢的原因在于需要遍历need去看是否满足条件，
        public String minWindow(String s, String t) {
            Map<Character, Integer> need = new HashMap<>();
            for (char c : t.toCharArray()) {
                need.put(c, need.getOrDefault(c, 0) + 1);
            }
            int l = 0, r = 0;
            int resLeft = -1, resLen = Integer.MAX_VALUE;
            int valid = 0;
            while (r < s.length()) {
                char c = s.charAt(r);
                if (need.containsKey(c)) {
                    need.put(c, need.get(c) - 1);
                    if (need.get(c) >= 0) valid++; //如果当前这个元素还需要，就给valid++
                }
                while (l <= r && valid == t.length()) {
                    int curLen = r - l + 1;
                    if (curLen < resLen) {
                        resLeft = l;
                        resLen = curLen;
                    }
                    char cl = s.charAt(l);
                    if (need.containsKey(cl)) {
                        need.put(cl, need.get(cl) + 1);
                        if (need.get(cl) > 0) valid--;
                    }
                    l++;
                }
                r++;
            }
            return resLeft == -1 ? "" : s.substring(resLeft, resLeft + resLen);
        }
    }

    static class normal_array {
        public int maxSubArray(int[] nums) {
            // 最大子数组和：前缀和？怎么确定数组哪两个元素差值最大
            // 贪心，要前面那部分还是只要当前元素，贪心更应该是通过每一步做出局部最优解最终得到全局最优解，类似于通过当前子组的和确定要不要当前这个元素，扩充子组长度
            // 应该是动态规划
            int pre = nums[0], res = nums[0];
            for (int i = 1; i < nums.length; i++) {
                pre = pre > 0 ? pre + nums[i] : nums[i];
                res = Math.max(res, pre);
            }
            return res;
        }

        public int[][] merge(int[][] intervals) {
            Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
            List<int[]> list = new ArrayList<>();
            for (int i = 0; i < intervals.length; i++) {
                int left = intervals[i][0];
                int right = intervals[i][1];
                while (i < intervals.length - 1 && right >= intervals[i + 1][0]) {
                    right = Math.max(right, intervals[i + 1][1]);
                    i++;
                }
                list.add(new int[]{left, right});
            }
            // list转array
            return list.toArray(new int[list.size()][2]);
        }

        // 轮转数组 : 翻转三次，或复制一个数组，然后num[(i+k)%n] = nums[i];

        // 除自身以外的数的乘积，不能用除法，同时除法要注意0
        // 没想出来，有点类似前缀和，一个逐步累积的思想
        public int[] productExceptSelf(int[] nums) {
            int[] res = new int[nums.length];
            // 不用Arrays.fill(res, 1);只赋值第一个就行，认为0左边所有元素乘积结果为1
            // 要掌握这种尽量不特殊处理边界值的技巧
            res[0] = 1;
            // 先计算当前元素左端所有元素的乘积
            for (int i = 1; i < nums.length; i++) {
                res[i] = res[i - 1] * nums[i - 1];
            }
            // 再计算当前元素右端所有元素的乘积
            int num = 1; // 比nums[nums.length-1]更好;
            for (int i = nums.length - 1; i >= 0; i--) {
                res[i] = res[i] * num;
                num *= nums[i];
            }
            return res;
        }

        public int[] productExceptSelf1(int[] nums) {
            // 双指针法，更酷
            int left = 1, right = 1;
            int[] res = new int[nums.length];
            Arrays.fill(res, 1);
            for (int i = 0, j = nums.length - 1; i < nums.length; i++, j--) {
                res[i] = res[i] * left;
                res[j] = res[j] * right;
                left *= nums[i];
                right *= nums[j];
            }
            return res;
        }

        // 缺失的第一个正数:没思路，没想到原地哈希,知道了原地哈希还是写成一坨
        public int firstMissingPositive(int[] nums) {
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] > 0 && nums[i] <= nums.length && nums[nums[i] - 1] != nums[i]) {
                    swap(nums, nums[i] - 1, i);
                    i--;
                }
            }
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] != i + 1) {
                    return i + 1;
                }
            }
            return nums.length + 1;
        }

        private void swap(int[] nums, int index1, int index2) {
            int temp = nums[index1];
            nums[index1] = nums[index2];
            nums[index2] = temp;
        }

    }

    static class matrix {
        // 旋转图像：写的吭吃瘪肚，1：遍历范围 2：映射关系
    }

    static class linked_list {
        // 相交链表：通过补链消除步行差，让双指针于交点相遇
        // 反转链表：递归迭代，递归反转链表会消耗额外空间
        // 回文链表: 法1：中间断开，翻转，比较；法2，把链表所有值存到一个list，然后左右指针遍历，比较值是否相等
        // 环形链表：快慢指针，扣圈; 也可看作判断遍历过程是否有节点重复出现，用set
        // 环形链表2:数学题，圈外长a,圈内长b，快慢指针，s1=2*s2 -> s1=s2+nb -> s2=nb,s1=2*nb 或 a+mb=2*(a+nb) -> a=(m-2n)b -> s1=(2m-2n)b
        //此时让指针三从head出发，指针一或二从当前位置出发，s3=a,s2'=s2+a=nb+a,s2与s3在入环处相遇
        // 合并两个有序链表:递归迭代,函数作用:合并两个有序链表,中止条件:有一个为null，无需合并
        // 两数相加
        // 删除倒数第n个节点
        // 两两交换链表中的节点
        // k个一组翻转链表
        class Node {
            int val;
            Node next;
            Node random;

            public Node(int val) {
                this.val = val;
                this.next = null;
                this.random = null;
            }
        }

        // 随机链表的复制：顺序复制想清楚需要什么，copy节点加到原节点后要注意细节
        // 重点：一次完成整个节点的复制：回溯深度搜和广度搜，递归写的不好，终止条件写的差
        Map<Node, Node> map = new HashMap<>();

        public Node copyRandomListDFS(Node head) {
            if (head == null) return null;
            if (map.containsKey(head)) {
                return map.get(head);
            }
            Node copy = new Node(head.val);
            // 没赋值next和random就放到map里
            map.put(head, copy);
            copy.next = copyRandomList(head.next);
            copy.random = copyRandomList(head.random);
            return copy;
        }

        public Node copyRandomList(Node head) {
            if (head == null) return null;
            //想清queue里存的节点是什么样的：一定是已经创建了带val的copy，但未必赋值了next和random
            Queue<Node> queue = new LinkedList<>();
            Node copyNode = new Node(head.val);
            queue.add(head);
            map.put(head, copyNode);
            while (!queue.isEmpty()) {
                Node cur = queue.poll();
                Node copy = map.get(cur);
                if (cur.next == null) {
                    copy.next = null;
                } else {
                    if (!map.containsKey(cur.next)) {
                        Node node = new Node(cur.next.val);
                        map.put(cur.next, node);
                        queue.add(cur.next);
                    }
                    copy.next = map.get(cur.next);
                }
                if (cur.random == null) {
                    copy.random = null;
                } else {
                    if (!map.containsKey(cur.random)) {
                        Node node = new Node(cur.random.val);
                        map.put(cur.random, node);
                        queue.add(cur.random);
                    }
                    copy.random = map.get(cur.random);
                }
            }
            return copyNode;
        }

        // 排序链表 ：最适合链表的排序是归并排序，如果要达到o(1)空间，需要采用自底向上的归并
        // 邪恶方法：读val到一个list，然后sort，然后遍历列表改值
        // 合并k个升序链表：归并的思路+合并两个有序链表
        // 其他：重复进行合并两个有序链表，复杂度极高；构造一个优先级队列，底层源码是小顶堆实现
        public ListNode mergeKLists(ListNode[] lists) {
            //<? super ListNode> 表示超类，包括当前类，当前类的父类，子类
            PriorityQueue<ListNode> pq = new PriorityQueue<>(new Comparator<ListNode>() {
                @Override
                public int compare(ListNode o1, ListNode o2) {
                    return o1.val - o2.val;
                }
            });
            if (lists == null || lists.length == 0) return null;
            for (ListNode list : lists) {
                if (list != null) pq.add(list);
            }
            ListNode dummy = new ListNode(0);
            ListNode pre = dummy;
            while (!pq.isEmpty()) {
                ListNode poll = pq.poll();
                pre.next = poll;
                if (poll.next != null) pq.add(poll.next);
                pre = pre.next;
            }
            return dummy.next;
        }
        // lru：数据结构还是没理清，方法定义还有待提高，细节处理有待提高
        // map直接存k-v？怎么把元素放到前面？加个list，如果k存在，需要o(1)定位节点，然后把节点放到最前面
        // 所以map要存 k-node。如何o(1)逐出最后一个，给一个tail，找tail的前一个。移动节点需要知道节点的前一个节点，所以需要双端队列
        // 方法：本题存在四种移动情况：1.get的移动(定位，remove，头插) 2.put原来有的(同get) 3.put原来没有的(头插) 4.超容量，移除最后一个

    }

    static class binary_tree {
        // tips:每道题都要思考递归的遍历和迭代的哪个更合理
//     * 36.中序遍历：递归写的还可以；迭代写的依托，总有死循环，要想好怎么解决根节点不能pop，同时while回来还会再访问这个节点
//        前序遍历，可以一层while，出栈根，入栈右左；可以两层while，只入栈右，出栈右；可以递归；可以morris
//        后序遍历，迭代很难
//                邪法：根右左前序，再reverse
//     * 37.二叉树深度:递归秒，可广度
//     * 38.翻转二叉树:递归秒，可广度
//     * 39.对称二叉树：注意判定方法，不是左子val=右子val就是对称的。注意迭代解法要会（广度）
//     * 40.二叉树直径:dfs
//     * 41.层序遍历打印:
//     * 42.有序数组转二叉搜索树：传递数组索引，注意细节
        public List<Integer> inorderTraversal1(TreeNode root) {
            Stack<TreeNode> stack = new Stack<>();
            List<Integer> list = new ArrayList<>();
            while (root != null || !stack.isEmpty()) { //外层while相当于完成一次左根右 //todo 区分前序，前序先把root放进栈，再while
                // 左根右，先左，所以要把root入栈
                while (root != null) { // while的作用在于找到当前root中第一个要被打印的节点；而前序第一个要打的就是root，所以只有一层while
                    stack.push(root);
                    root = root.left;
                }
                root = stack.pop(); // 栈里只有根，所以出的也是根
                list.add(root.val);
                root = root.right; // 再看右
            }
            return list;
        }

        public List<Integer> inorderTraversal(TreeNode root) {
            List<Integer> list = new ArrayList<>();
            while (root != null) {
                if (root.left != null) {
                    TreeNode predecessor = root.left;
                    while (predecessor.right != null && predecessor.right != root) {
                        predecessor = predecessor.right;
                    }
                    if (predecessor.right == root) {
                        predecessor.right = null;
                        root = root.right;
                        list.add(root.val);
                    } else {
                        predecessor.right = root;
                        root = root.left;
                    }
                } else {
                    list.add(root.val);
                    root = root.right;
                }
            }
            return list;
        }

        //     * 43.验证二叉搜索树:想到了迭代中序遍历，创建一个变量表示前驱
//            又认为递归做不了，实际能做，也是中序遍历，全局变量记录前一个值；更没想到前序遍历也能做，每个节点给一个界，树种所有节点都符合自己的界就是搜索树
//     * 44.二叉搜素树中第k小的元素:上来先写了一个递归的中序遍历，确实做出来了，但实际上本题应该是迭代的遍历更好
//            也想到了取巧的方法，打值到数组里，然后出结果
//            如果搜索树经常修改，如何找第k小的元素：优先级队列，维护大小为k的大顶堆，比堆顶小就入堆 todo 如何建堆
//            方法2：（没想到）维护树的节点个数,可以自定义一个数据结构或是维护一个全局map，全局map更好
//     * 45.二叉树的右视图:bfs扫一遍，取每层最后一个即可
//              法:深度搜也能做，因为要从上到下的右视图，所以根右左，遍历的同时传递深度参数，深度大于res.size就加个新的
//     * 46.二叉树展开为链表：要意识到是一个根左右的遍历，同时把当前节点和前一个节点连接就很好做，但要注意这个过程中右节点的丢失
//            第三个核心点在于展平过程会改变树的结构，如flatten(root.left)再flatten(root.right)会栈溢出
//            可以产生多种解法，借助全局pre的递归(前序，root连上去，递归左，递归右，要记得保存右) 迭代(前序,一次出一个，把自己连到pre上，右左入)，
//            不借助pre的递归（返回展平后的最后一个节点） 迭代（两层while，把右入栈，左连过来，一直到没有左，直接向右或出栈一个连过来）
//            头插法，借助全局pre，按照右左根的顺序把，pre连到当前节点上，因为执行完root.right之后，root.left的指向没变，所以不需要栈或变量额外表村
//            如果要求空间就只能前序morris，左子树最右连到右子树


        //     * 47.从前序与中序遍历序列构造二叉树:很熟了，不用在写了
//     * 48.路径总和3：不熟，关键在于如何找出所有路径。两个解法都借助map，保存了包含当前节点的全部路径，但前者是从下到上，后者是从上到下
//              写出一个左右根的解法，好理解但耗空间，每个节点返回一个map，里面储存了一定包含该节点，能凑出来的key，每个k有v种凑法
//              根左右的解法实际上更好，但更难理解，用一个前缀和的思想，map只存一条根节点到当前节点的路径信息，切换分支map要还原，相当于每到一个节点计算以他为结尾的成功路径个数
//              解法3：双层dfs，第一层根左右遍历所有起点，第二层接收起点遍历子树终点
        Map<Long, Integer> map = new HashMap<>();

        public int pathSum(TreeNode root, int targetSum) {
            this.target = targetSum;
            map.put(0L, 1);
            preOrderPathSum(root, 0);
            return resPathSum;
        }

        int resPathSum = 0;
        int target;

        public void preOrderPathSum(TreeNode root, long sum) {
            // 传入一个节点，返回
            if (root == null) return;
            //前序遍历，向水流往下流，从根到当前节点为一条路径
            // 维护一个map，储存了根节点到该路径上其他节点组成的子路径
            // 通过这个map可以计算出以某个节点为结尾，从上倒下的所有路径之和
            sum += root.val;
            // 先put再算res：如果target=0，就会多算一个
            // 前缀有多少种方式凑出target-sum，以当前节点为结尾就有多少方式凑出target
            resPathSum += map.getOrDefault(sum - target, 0);
            map.put(sum, map.getOrDefault(sum, 0) + 1);
            preOrderPathSum(root.left, sum);
            preOrderPathSum(root.right, sum);
            // 重点,返回上一层递归，之后从左枝移到右枝，所以要在返回之前，还原map，因为map中存的是一条分支中的前缀路径信息
            map.put(sum, map.get(sum) - 1);
        }

        //     * 49.二叉树的最近公共祖先，难点在递归函数的设计
//        写出一个垃圾方法，写一个函数，判断node属不属于root所在的树，返回一个bool，然后if，例如：左子右子都返回true，那答案一定是root
//        法：递归函数判断root中是否存在pq之一，后续遍历，if判断是否满足最近根节点条件（唯一）(root为pq之一且左或右存在pq之一；左存在p右存在q)
//        法：前序遍历，如果root等于p或q，那么res一定是root或root更上层。本质上是找pq并返回可能的解
//           递归函数：返回可能的最近父节点，并再之后验证确定该继续返回还是返回一个新的；如果一定不是，返回null
        // 法：map储存每个节点的父节点，从p开始往上找，把到root的路径存入set，再从q往上找，碰到出现过的就返回
//     * 50.二叉树最大路径和，很熟了不用写了，后续遍历，递归函数计算以当前节点起始向下的最大路径和，全局变量整合通过该节点的最大路径
        public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
            Set<TreeNode> set = new HashSet<>();
            preOrderLowesCom(root);
            anceMap.put(root, null);
            TreeNode node = p;
            while (node != null) {
                set.add(node);
                node = anceMap.get(node);
            }
            node = q;
            while (node != null) {
                if (set.contains(node)) return node;
                node = anceMap.get(node);
            }
            return root;
        }

        Map<TreeNode, TreeNode> anceMap = new HashMap<>();

        public void preOrderLowesCom(TreeNode root) {
            if (root == null) return;
            if (root.left != null) anceMap.put(root.left, root);
            if (root.right != null) anceMap.put(root.right, root);
            preOrderLowesCom(root.left);
            preOrderLowesCom(root.right);
        }

        // 传入一个节点，判断pq之一是否在其中
        TreeNode judgeRes = null;

        public boolean judgeLowestCom(TreeNode root, TreeNode p, TreeNode q) {
            if (root == null) return false;
            // 递归函数判断是否存在pq之一在该树中
            // 希望尽可能深，所以是后续
            boolean leftContains = judgeLowestCom(root.left, p, q);
            boolean rightContains = judgeLowestCom(root.right, p, q);
            // 最近根节点一定是pq分别在左右 或者 其中一个等于root，另一个存在于left或right，且只有最近的会满足这两个条件
            if (leftContains && rightContains) {
                judgeRes = root;
            }
            if ((leftContains || rightContains) && (root.val == p.val || root.val == q.val)) {
                judgeRes = root;
            }
            return leftContains || rightContains || root.val == p.val || root.val == q.val;
        }

        public TreeNode lowestCAncestor(TreeNode root, TreeNode p, TreeNode q) {
            // 递归函数作用，找到pq并把他返回回来，如果发现左右都找到了，就返回root
            // 如果两个都有返回，代表左边看见一个，右边看见一个，返回root，代表root才是真正的解
            // 如：左枝找到p返回，去另一枝找q，没找到把p向上返回，再去上层的另一枝找q；找到了，返回该层root，没找到继续向上去另一枝找q。
            if (root == null) return null;
            // 先看根
            // 最近祖先一定是root或root更上层
            if (root == p || root == q) return root;
            // 再看左右，返回的lNode和rNode可能为最近祖先
            TreeNode lNode = lowestCAncestor(root.left, p, q);
            TreeNode rNode = lowestCAncestor(root.right, p, q);
            // 再次验证，如果l和r都有返回值，那么返回root
            if (lNode != null && rNode != null) return root;
            if (lNode != null) return lNode;
            if (rNode != null) return rNode;
            // 左右都是null说明pq不在此树，返回null
            return null;
        }

    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        int[] res = new int[nums.length - k + 1];//起始：0到等于n-k
        Deque<Integer> deque = new LinkedList<>();//维护的是一个单调非增队列
        //后入,如果前面的元素比入的还小，那他就不可能是最大值，因为窗口中一定包含最后入的这个元素
        //怎么判断队头已经到时间了，需要出队了,根据deque中存的下标
        for (int i = 0; i < nums.length; i++) {
            while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {//如果相等也会入队
                deque.pollLast();
            }
            deque.addLast(i);
            //计算以i结尾的窗口的最大值
            if (deque.peekFirst() == i - k) deque.pollFirst();
            if (i >= k - 1) res[i - k + 1] = nums[deque.peekFirst()];
        }
        return res;
    }

    public void rotate(int[] nums, int k) {
        //怎么达到o（1）空间？三次翻转，没想到，一个整体的思想
        //nums[i] -> nums[i+k%n]
        int n = nums.length;
        k = k % n;
        int count = 0;
        int start = 0;
        //跳着改变元素，count记录已经改变的个数
        while (count < n) {
            int curIndex = start;
            int prev = nums[start];
            do {
                int nextIndex = (curIndex + k) % n;
                int temp = nums[nextIndex];
                nums[nextIndex] = prev;
                count++;
                prev = temp;
                curIndex = nextIndex;
            } while (curIndex != start);//必须do while 否则第一次进不了循环
            start++;
        }
    }

    public int[] productExceptSelf(int[] nums) {
        //不让用除法，而且除法要注意可能有0
        //法1：开两个数组分别储存左积和右积---法2：直接使用res，先从左到右记录左积，再从右到左把右积乘进去----法3双指针，同时进行左到右和右到左
        int n = nums.length;
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1; i < n; i++) {
            res[i] = res[i - 1] * nums[i - 1];
        }
        int r = 1;
        for (int j = n - 1; j >= 0; j--) {
            res[j] = res[j + 1] * r;
            r = r * nums[j];
        }
        return res;
    }

    public int firstMissingPositive(int[] nums) {
        //如何o(n)+o(1)
        //可能是缺个大的 1235，也可能是缺个小的234
        //1.存哈希表，然后从1开始枚举，看表里有没有 o(n)+o(n)
        //2.不用哈希表，从1枚举，看数组有没有 o(n^2)+o(1)
        //重点：res已经是[1,N+1]中的数
        for (int i = 0; i < nums.length; i++) {
            //把nums[i]换到下标num[i-1]的位置
            //换过来的元素也需要处理
            while (nums[i] > 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
            }
        }
        //1出现了第一个位置一定被换成1，找到第一个不符合规则的就是缺失的最小正数
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) return i + 1;
        }
        return nums.length + 1;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int i = 0, j = n - 1;//从右上或左下开始搜索都行，重点在于两个方向一个变小一个变大，从而一点点减小范围
        while (i < n && j >= 0) {//注意一定不能是三个if，第二个if完成后会进到第三个
            if (matrix[i][j] == target) return true;
            else if (matrix[i][j] > target) j--;
            else i++;
        }
        return false;
    }

    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode temp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = temp;
        }
        return prev;
    }

    public boolean isPalindrome(ListNode head) {
        //法1：额外空间，存到数组里 法2：后半反转
        ListNode slow = head, fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode head2 = slow.next;
        slow.next = null;
        head2 = reverseList(head2);
        while (head2 != null) {
            if (head.val != head2.val) return false;
            head = head.next;
            head2 = head2.next;
        }
        return true;
    }

    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode(0, head);
        ListNode pre = dummy;
        ListNode cur = head;
        while (cur.next != null) {
            ListNode temp = cur.next.next;
            pre.next = cur.next.next;
            cur.next.next = cur;
            cur.next = temp;

            pre = cur;
            cur = pre.next;
        }
        return dummy.next;
    }

    //36 中序遍历，递归的时候隐式地维护了一个栈，而我们在迭代的时候需要显式地将这个栈模拟出来
    //还有其他写法，比如前序遍历，可以把右节点入栈，也可以右左入栈；比如中序遍历，可以右根入栈
    //后续怎么遍历，什么时候打印值？法1：逆后续，根右左的顺序遍历，然后reverse
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();//栈里放的是根，用根去找右
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.push(root);//
                root = root.left;//再往左移
            }
            root = stack.pop();
            res.add(root.val);
            root = root.right;
        }
        return res;
    }

    //38
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return root;
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = null;
        // 对照递归来想，塞进去执行left，拿出来准备执行right再塞进去，再拿出来打印val；所以要区分拿出来之后是找right还是打印值
        // 根入找左 - 根出找右，右没访问过，再跟入 - 根出，左右已经访问过，打印根值
        while (!stack.isEmpty() || root != null) {
            // 1.左
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            // 2.没左，出根找右
            root = stack.pop();
            TreeNode right = root.right;
            if (right == null || right == prev) { // 右子树为空或者右子树已经被访问过就打根
                //相当于左右都看完了，1.打根值 2.出栈上一层的根，然后去找右，所以要回到第二部
                list.add(root.val);
                prev = root;
                root = null; // root已经打印过了，为了让他移动，进入循环开始一定要是一个新的root；同时置为null意味着左子树访问过了
            } else { // 有右子树，需要把右子树全处理完，所以需要回到第一部
                // 但是root出栈又入栈一定会导致死循环，怎么判定，用一个prev，如果他的右子树的值已经被打过了，那么就直接打他的值
                stack.push(root);
                root = right; // 准备处理右子树
            }
        }
        return list;
    }
    public void post(TreeNode root,List<Integer> list){
        if(root==null) return;
        post(root.left,list);
        post(root.right,list);
        list.add(root.val);
    }

}
