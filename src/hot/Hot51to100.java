package hot;

import java.beans.Visibility;
import java.util.*;

public class Hot51to100 {
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

    /**
     * 二分查找


     * 67.寻找旋转排序数组中的最小值
     * 68.寻找两个正序数组的中位数，想的不够准确，实现也没有实现出来，把本题当成找两个有序数组的第k个数
     * 栈
     * 69.有效的括号
     * 70.最小栈:做了很多次了
     * 71.字符串解码：栈的思想，类似逆波兰表达式
     * 72.每日温度:单调栈，找右侧比当前元素大的第一个元素
     * 73.柱状图中最大的矩形：先想枚举，暴力解法必须会
     */
    static class graph_theory {
        // 前置：图的深度和广度遍历
        List<Integer> list = new ArrayList<>();

        // 从任意位置开始，遍历整个二维矩阵（注意不是遍历所有顶点）
        public void depthFirst(char[][] grid, int i, int j) {
            if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != '1') {
                return;
            }
            grid[i][j] = '0'; //一定有，否则会栈溢出；本质上是为了标记已经访问过
            depthFirst(grid, i - 1, j);
            depthFirst(grid, i + 1, j);
            depthFirst(grid, i, j - 1);
            depthFirst(grid, i, j + 1);
        }

        //      * 51.岛屿数量，dfs;为了防止死循环要有visited数组，或者直接在原数组改值，表示当前路径上已经访问过该节点
//             或bfs;bfs写的有问题(重复入队导致超时，应该在入队之前置0，而非出队置0，防止看到同一个位置二次入队)
//             或并查集;找到一个土，把这个土上下左右的土合并到这块土所属的岛上
//     * 52.腐烂的橘子：正法：多源广度优先搜索，细节有问题，如果已经没有新鲜的了，还会被再腐一次
//             或dfs(不好)，可以从1出发找最近的2；也可以从2出发，计算腐烂每个1要多久，然后保留不同2中的最短用时；同样，为了防止死循环，要有个visited数组标记
//     * 53.课程表：核心是判断有向图是否有环，bfs（入度为0的出队，削减其他节点的入度）
//             或dfs（三色标记）,先构建邻接表会减少很多时间
//      * 54.实现前缀树:数据结构定义没整明白，细节有问题
        public int numIslands(char[][] grid) {
            int m = grid.length;
            int n = grid[0].length;
            UnionFind uf = new UnionFind(m * n);
            int zeroCount = 0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (grid[i][j] == '0') zeroCount++;
                    if (grid[i][j] == '1') {
                        // 找到一个新的土，把周围的土合并到这块土所属的岛上
                        grid[i][j] = '0';
                        if (j - 1 >= 0 && grid[i][j - 1] == '1') {
                            uf.union((i) * n + j, (i) * n + j - 1);
                        }
                        if (j + 1 < n && grid[i][j + 1] == '1') {
                            uf.union((i) * n + j, (i) * n + j + 1);
                        }
                        if (i - 1 >= 0 && grid[i - 1][j] == '1') {
                            uf.union((i) * n + j, (i - 1) * n + j);
                        }
                        if (i + 1 < m && grid[i + 1][j] == '1') {
                            uf.union((i) * n + j, (i + 1) * n + j);
                        }
                    }
                }
            }
            int count = 0;
            for (int i = 0; i < m * n; i++) {
                if (i == uf.find(i)) count++;
            }
            return count - zeroCount;
        }

        // 本质是判断有没有环？
        // 广度搜，找所有入度为0的节点，并削减下一层节点的入度
        public boolean canFinish(int numCourses, int[][] prerequisites) {
            // 构建邻接表
            List<List<Integer>> adjacency = new ArrayList<>(); //用Map<Integer,List<Integer>>也行
            for (int i = 0; i < numCourses; i++) {
                List<Integer> list = new ArrayList<>();
                adjacency.add(list);
            }
            // 构建入度矩阵
            int[] indegree = new int[numCourses];
            for (int[] prerequisite : prerequisites) {
                // 学完p[1]才能学p[0]，课程i指向p[0],i的边表中加入p[0]
                adjacency.get(prerequisite[1]).add(prerequisite[0]);
                // 课程i有一个前置，入度+1
                indegree[prerequisite[0]]++;
            }
            // 广度优先搜索，先处理入度为0的
            Queue<Integer> queue = new LinkedList<>();
            for (int i = 0; i < numCourses; i++) {
                if (indegree[i] == 0) queue.add(i);
            }
            while (!queue.isEmpty()) {
                // 取出入度为0的课程，代表可以学，并把其后置课程的入度-1
                int course = queue.poll();
                List<Integer> list = adjacency.get(course);
                for (Integer i : list) {
                    indegree[i]--;
                    if (indegree[i] == 0) queue.add(i);
                }
            }
            for (int i : indegree) {
                if (i != 0) return false;
            }
            return true;
        }

        // 三色标记法
        public boolean canFinish1(int numCourses, int[][] prerequisites) {
            // 构建邻接表
            List<List<Integer>> adjacency = new ArrayList<>();
            for (int i = 0; i < numCourses; i++) {
                List<Integer> list = new ArrayList<>();
                adjacency.add(list);
            }
            for (int[] prerequisite : prerequisites) {
                // 学完p[1]才能学p[0]，课程i指向p[0],i的边表中加入p[0]
                adjacency.get(prerequisite[1]).add(prerequisite[0]);
            }
            int[] colors = new int[numCourses]; // 三色标记，0表示没访问过，1表示在当前访问的路径，2表示已经追溯到最后发现该课程可以学
            for (int i = 0; i < numCourses; i++) {
                if (hasCycle(adjacency, i, colors)) return false;
            }
            return true;
        }

        public boolean hasCycle(List<List<Integer>> adjacency, int course, int[] colors) {
            if (colors[course] == 2) return false;
            if (colors[course] == 1) return true;
            colors[course] = 1; //1不用再置回为0，因为如果无环，执行完成后，整条路径的1一定都会置为2；有环最后会返回false，结束执行，也就不关心colors中的1了
            for (Integer i : adjacency.get(course)) { //看前置课程能不能学
                // 从i往上走，发现有环，则course学不了(course往上走有环)
                if (hasCycle(adjacency, i, colors)) return true;
            }
            // 前置课程全能学，该课程已判断完毕
            colors[course] = 2;
            return false;
        }


        class Trie {
            // 多级数组嵌套怎么实现 -> 数组中的每个位置储存指向下一个数组的指针
            // 指针怎么实现？想二叉树，所以自定义Node，有一个属性是Node[]
            class Node {
                private Node[] next; //如果用private修饰，就是封装那一套，要写set/get才能被类外面的代码读取;但他是内部类，外面的类中所有的代码仍然能对象.属性
                private boolean isEnd;

                public Node() {
                    this.isEnd = false;
                    this.next = new Node[26];
                }

            }

            private Node root; // 注意

            public Trie() {
                this.root = new Node();
            }

            public void insert(String word) {
                Node cur = root;
                for (char c : word.toCharArray()) {
                    // a = new node[]只创建了一个容器。调用a[0]返回null
                    Node[] next = cur.next;//Node[] next = root.getNext(); 不用get
                    if (next[c - 'a'] == null) {
                        next[c - 'a'] = new Node();
                    }
                    cur = next[c - 'a'];
                }
                cur.isEnd = true;
            }

            public boolean search(String word) {
                Node cur = root;
                for (char c : word.toCharArray()) {
                    Node[] next = cur.next;
                    if (next[c - 'a'] == null) {
                        return false;
                    }
                    cur = next[c - 'a'];
                }
                return cur.isEnd;
            }

            public boolean startsWith(String prefix) {
                Node cur = root;
                for (char c : prefix.toCharArray()) {
                    Node[] next = cur.next;
                    if (next[c - 'a'] == null) {
                        return false;
                    }
                    cur = next[c - 'a'];
                }
                return true;
            }
        }

    }

    static class backtracking {
        //     * 55.全排列：秒了，多叉树基本回溯(撤销上一个，换个新的)
//     * 56.子集：要或不要：秒；
//              start多叉树(回溯模板)没想明白；
//              二进制、动态规划没想到，二进制写不出来；
//     * 57.电话号码的数字组合:秒了，多叉树基本回溯，剪枝都没有
//     * 58.组合总和：数组回溯模板(一定要会)，
//              选或不选，没啥区别，看curTarget<0或start==num.len终止
//              完全背包:一般只要个结果的采用动态规划，要具体方案的采用回溯
//     * 59.括号生成：秒了，基本回溯，想清非法和合法的条件就行。进阶：区分括号种类，除了考虑数目，还要考虑合法闭合，所以加个stack
//     * 60.单词搜索：秒了，基本回溯，就是图的遍历，可不用visited选择，修改为“ ”节省空间
//     * 61.分割回文串：基本回溯，忽略了用动态规划预处理或搜索过程中的记忆化 减少回文串的判断耗时
//     * 62.N皇后：还真写出来了，关键在于理解题意，找到一个符合条件的col排序，条件怎么表述？同一列不行：set，对角线不行：左上右下：i-j不同；右上坐下：i+j不同
        List<List<Integer>> res = new ArrayList<>();

        public List<List<Integer>> subsets(int[] nums) {
            // [] - 1 - 12 -123 -13 -2 -23 -3
            // 空 - 一定选1 - 一定选1+（空/一定选2+（空/一定选3）/一定选3） - 一定选2+（空/一定选3） - 一定选3
            // 同层：先一定选1，一定不选1且一定选二2，一定不选12且一定选3
            // 下层：一定选1的基础上：一定啥也不选，一定选2，一定不选2且一定选3
            //      一定选2的基础上：一定啥也不选/一定选3
            // start的作用本质在于去重，重复是在以不同顺序选择元素获得集合时产生的，先选一再选二和先选二再选一是一样的
            // 对应上面就是：一定选1后，同时考虑了一定选2和一定不选2；所以同层中，一定选2后，就不该再考虑一定选1
            subsets(nums, 0, new ArrayList<>());
            return res;
        }

        public void subsets(int[] nums, int start, List<Integer> path) {
            // 全集 = 一定啥也不选 (1)+一定先选个1 (1+2+1)+一定先选个2 (2)+一定先选个3 (1) = 8
            res.add(new ArrayList<>(path));
            for (int i = start; i < nums.length; i++) {
                path.add(nums[i]);
                subsets(nums, i + 1, path);
                path.remove(path.size() - 1);
            }
        }

        public List<List<Integer>> combinationSum(int[] candidates, int target) {
            List<List<Integer>>[] dp = new List[target + 1]; // 凑出j的具体全部方案，每种方案是一个List<Integer>，多种方案就是List<List<Integer>>
            for (int i = 0; i < dp.length; i++) {
                dp[i] = new ArrayList<>();
            }
            dp[0].add(new ArrayList<>());
            // 完全背包
            for (int i = 0; i < candidates.length; i++) {
                // 用前i个元素有多少种方案能凑出j
                for (int j = candidates[i]; j <= target; j++) {
//                    dp[i][j] = dp[i-1][j]+dp[i][j-nums[i]] -> dp[j] = dp[j] + dp[j-nums[i]];
//                    用前i个元素凑出j的所有方案 = 用前i-1个元素凑出j的所有方案 + 用前i个元素凑出j-num[i]的所有方案再拼接num[i]
                    //                       = dp[j]                   + dp[j-num[i].copy.AllAdd(num[i])
                    List<List<Integer>> lists = dp[j - candidates[i]];
                    for (List<Integer> list : lists) {
                        List<Integer> copy = new ArrayList<>(list);
                        copy.add(candidates[i]);
                        dp[j].add(copy);
                    }
                }
            }
            return dp[target];
        }
//        public List<List<Integer>> combinationSum(int[] candidates, int target) {
//            Arrays.sort(candidates); // 排序，用于后面curtarget<0的剪枝
//            combination(candidates, target, new ArrayList<>(), 0);
//            return res;
//        }

        public void combination(int[] cadidates, int curTarget, List<Integer> path, int start) {
            // 全排列的思想必然实现全覆盖，但存在大量重复，得到结果后还需去重
            // 不考虑顺序的问题，按某种顺序搜索，实现去重
            // 如果按全排列的思想：会产生 先选1再选2和先选2再选1 但他们没有区别，都需要再凑target-3
            // 通过每一轮搜索设置下一轮搜索的起点实现去重，即start的意义
            if (curTarget == 0) {
                res.add(new ArrayList<>(path));
                return;
            }
            if (curTarget < 0 || start == cadidates.length) return;
            for (int i = start; i < cadidates.length; i++) {
                path.add(cadidates[i]);
                combination(cadidates, curTarget - cadidates[i], path, i);
                path.remove(path.size() - 1);
            }
        }
        // 59 括号生成
        // 进阶，如果有多种括号该怎么做，即除了数量，还要保证嵌套类型合法，如([)] 是非法的，而 ([]) 是合法的
        // 解决：参数传递加一个stack，里面存左括号，然后向builder中添加左括号无条件，添加右括号要保证与栈顶的括号类型匹配，出栈，再进入下一层递归，回溯时栈也要回溯

        public boolean exist(char[][] board, String word) {
            //如何剪枝优化：使其在 board 更大的情况下可以更快解决问题？
            // 剪枝本质上就是及时发现此路不通，越早越好
            // 所以索引越界，已访问过，当前字符与目标字符不同 这些判断导致的return就叫剪枝
            // 进一步(了解即可)，因为剪枝越早越好，所以可以判断，word中第一个字符和最后一个字符在board中出现的次数，如果最后一个字符出现的少，就把word倒过来搜
            // 统计word中字符出现的次数，如果某个字符在board中出现的次数小于word中出现的次数，一定false
            // 可以用int[128]统计(ascall)，for(char c:word) int[c]++
            boolean[][] visited = new boolean[board.length][board[0].length];
            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j < board[0].length; j++) {
                    if (dfs(board, visited, word, i, j, 0)) return true;
                }
            }
            return false;
        }

        public boolean dfs(char[][] board, boolean[][] visited, String word, int x, int y, int index) {
            // 空间优化，可以不用visited，因为进入下层递归说明当前board[x][y] = word.charAt(index)
            // 可以将board[x][y]先置 " ",然后回溯再通过word还原
            if (index == word.length()) {
                return true;
            }
            if (x < 0 || x >= board.length || y < 0 || y >= board[0].length ||
                    visited[x][y] ||
                    board[x][y] != word.charAt(index)) {
                return false;
            }
            visited[x][y] = true;
            boolean res = dfs(board, visited, word, x + 1, y, index + 1) ||
                    dfs(board, visited, word, x - 1, y, index + 1) ||
                    dfs(board, visited, word, x, y + 1, index + 1) ||
                    dfs(board, visited, word, x, y - 1, index + 1);
            visited[x][y] = false;
            return res;
        }


    }

    static class UnionFind {
        int[] parent;
        int[] rank;

        public UnionFind(int n) {
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                rank[i] = 1;
            }
        }

        public void union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX == rootY) return;
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++; // 如果两棵树的深度相同，合并后树的深度增加1
            }
        }

        public int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }

        //集，判断x和y是否属于同一集合，集需要查找根；合并也需要用到查找跟，只要是查根就用find(i)而非parent[i]
        public boolean isConnected(int x, int y) {
            return find(x) == find(y);
        }

    }

    static class binary_search{
//     * 63.搜索插入位置：题意理解了，二分没写明白，四个模板，加个result就好理解不少，找最后一个严格小于t的，最后一个小于等于t的，第一个严格大于t的，第一个大于等于t的
//       第一个严格小于t的(没意义，直接看0就行)，最后一个大于t的(直接看n-1就行)
//        todo :多个相等元素，第一个等于t的和最后一个等于t的
//     * 64.搜索二维矩阵：三种方法，左下向右上O(m+n)，先行二分再列二分O(logm+logn)，展平成一维O(logmn)
//     * 65.在排序数组中查找元素的第一个和最后一个位置：两次二分，找第一个大于等于target的和最后一个小于等于target的
//     * 66.搜索旋转排序数组：主要还是细节
        public int searchInsert(int[] nums, int target) {
            // 补：用例上看，题意要求从前往后插，[1,3,5,6]，target=5，要求返回2
            // 找最后一个严格小于t的 或 第一个大于等于t的
            int left=0,right = nums.length-1;
            int result = -1;// 从第一个严格小于t的位置，向右找，找到最后一个严格小于t的
            while (left<=right){
                int mid = (left+right)/2;
                //一定是mid-1和mid+1，否则没办法收敛，可通过，只有两个元素时，或只有一个元素时，发生死循环，来验证
                if(nums[mid]<target){ //找什么这个位置的条件就是什么，result初始化都是-1
                    result = mid;// 记录当前索引
                    left = mid+1;// 向右继续查找
                }else{
                    // right指向一个可能的解；因为mid是大于等于t的，一定不是解，所以再向内移动一步
                    right = mid-1;
                }
            }
            // right是最后一个严格小于t的位置
            return result+1;
            //第一个大于等于t的
//            while (left<=right){
//                int mid = (left+right)/2;
//                if(nums[mid]>=target){
//                    result = mid
//                    right = mid-1;
//                }else{
//                    left = mid+1;
//                }
//            }
//            return result;
        }
        public int[] searchRange(int[] nums, int target) {
            int left=0,right = nums.length-1;
            int lRes = -1;
            int rRes = -1;
            while (left<=right){
                int mid = (left+right)/2;
                if(nums[mid]==target){
                    lRes = mid;
                    right = mid-1;
                }else if(nums[mid]>target){
                    right = mid-1;
                }else {
                    left = mid+1;
                }
            }
            left=0;right = nums.length-1;
            while (left<=right){
                int mid = (left+right)/2;
                if(nums[mid]==target){
                    rRes = mid;
                    left = mid+1;
                }else if(nums[mid]>target){
                    right = mid-1;
                }else {
                    left = mid+1;
                }
            }
            return new int[]{lRes,rRes};
        }
        public int findMin(int[] nums) {
            // 没啥思路：折线图想到了
            // 如果当前无序，最小值一定在无序的半区？如果有序，取nums[left]?
            int left = 0,right = nums.length-1;
            while (left<=right){
                int mid = (left+right)/2;
                // 本质上就是三种情况 1.直线 2.mid在闪电左边 3.mid在闪电右面
                //当前[left,right]为直线(升序)
                if(nums[left]<=nums[mid] && nums[mid]<=nums[right]){
                    return nums[left];
                } else { // 不是直线，实际是就是mid在闪电的左边还是右边
                    if (nums[left]<=nums[mid]) { //mid在左边，[left,mid]是直线，解一定在mid右边
                        left = mid+1;
                    }else { // mid落在右边 right = mid-1
                        // 为什么不是mid-1  [3,1,2] ，mid落在缝里，mid可能是解
                        // 不-1能保证收敛吗？能，因为是right，区间一定是越来越窄的
                        right = mid;
                    }
                }
            }
            return -1;
        }

    }

    
    //旋转排序数组最小值
    public int findMin(int[] nums) {
        //区分旋转排序数组，因为最小值不一定在有序部分
        //比mid和right
        int left = 0, right = nums.length - 1;
        while (left <= right) {//等于？
            int mid = (left + right) / 2;
            if (nums[mid] < nums[right]) {//等于？
                right = mid;//减1？不能减，mid可能是答案,可不减的话最后剩两个元素，left如果是答案就会一直循环？不会循环，因为两个数的话mid是左
            } else {//if(mid>=right)
                left = mid + 1;
            }
        }
        return nums[right];//left?right?
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        //第一反应：两个数组都定位到一半的位置，把较小的那个所在的一半移除，因为一共需要移除m/2+n/2
        //为什么不对？没有站在一个把两个数组当成整体的角度来考虑，应该要找全局第k个，然后分别从两个数组取k/2然后比较
        int m = nums1.length, n = nums2.length;
        int left = (m + n + 1) / 2;
        int right = (m + n + 2) / 2;
        return (getKth(nums1, nums2, 0, 0, left) + getKth(nums1, nums2, 0, 0, right)) * 0.5;
    }

    public int getKth(int[] nums1, int[] nums2, int i1, int i2, int k) {
        if (i1 == nums1.length) {
            return nums2[i2 + k - 1];
        }
        if (i2 == nums2.length) {
            return nums2[i1 + k - 1];
        }
        if (k == 1) {
            return Math.min(nums1[i1], nums2[i2]);
        }
        //if(nums1[i1+k/2-1]>nums2[i2+k/2-1]) 可能越界，没考虑到
        int p1 = Math.min(nums1.length - 1, i1 + k / 2 - 1);
        int p2 = Math.min(nums2.length, i2 + k / 2) - 1;
        if (nums1[p1] < nums2[p2]) {
            //把更小的所在的数组的那部分k/2移除，能保证第k个数依然被留下来
            return getKth(nums1, nums2, p1 + 1, i2, k - (p1 - i1 + 1));
        } else {
            return getKth(nums1, nums2, i1, p2 + 1, k - (p2 - i2 + 1));
        }
    }

    public boolean isValid(String s) {
        //如果结构有要求怎么办   {[()()],[]},{()}
        //记录上一个闭合的括号，当前闭合中的话上一个不能是大{}？不对当前闭合大上一个不能是中[]，闭合一个大要清除之前的状态 []{}和{[]}
        return false;
    }

    public String decodeString1(String s) {
        //因为涉及到嵌套，先处理内层，再处理外层，所以是栈
        //1.一定要弄清楚栈里存的是什么，有两种解法，1.字母一个栈数字一个站 2.全用一个栈
        // 本质上是一个消去括号的过程，读到 [ 新开一个sb，存之后的字符，读到]，拿到[前面的第一个数字（离当前这组括号最近的数字），得到重复次数，再从sb栈中取出上一组括号内的sb和这个拼接，就得到上一组括号内的全部内容
        //3[a2[cd]] 读a--读2--读[--把2入栈--开一个sb，拼接cd--读到]--栈里取出2--2份cd--栈里取出a，拼接2份cd
        StringBuffer res = new StringBuffer();
        Stack<Integer> numStack = new Stack<>();
        Stack<StringBuffer> sbStack = new Stack<>();
        int num = 0;
        for (char c : s.toCharArray()) {
            //为什么不先是if(Character.isLetter(c))
            //1.是左括号，把res，前一个数字入栈,重置res记录接下来的字符
            if (c == '[') {
                numStack.push(num);
                num = 0;//别忘，否则算下一个括号前的num会出错
                sbStack.push(res);
                res = new StringBuffer();
            }
            //2.是右括号，出栈次数，res乘以次数，和之前的res拼接
            else if (c == ']') {
                StringBuffer temp = new StringBuffer();
                int preNum = numStack.pop();
                for (int i = 0; i < preNum; i++) {
                    temp.append(res);
                }
                res = sbStack.pop();
                res.append(temp);
            }
            //3.是数字，记录这个数字，用于之后遇到左括号时入栈，为什么一定要入栈？防止嵌套丢失信息
            else if (c >= '0' && c <= '9') {//[]前的数字可能是多位数
                num = c - '0' + num * 10;
            }
            //4.其他
            else {
                res.append(c);
            }
        }
        return res.toString();
    }

    //一个栈更好理解了，但是需要翻转字符串或者调用insert头插
    public String decodeString2(String s) {
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()) {
            if (c != ']') {//除了]全入栈
                stack.push(c);
            } else {//读到右括号，一直出到与之配对的左括号，然后完成一组拼接，再入栈
                StringBuffer sb = new StringBuffer();
                while (!stack.isEmpty() && stack.peek() != '[') {
                    sb.insert(0, stack.pop());//头插，否则需要反转
                }
                String subString = sb.toString();//当前括号内的内容
                //出[
                stack.pop();
                //获取num
                sb = new StringBuffer();
                while (!stack.isEmpty() && Character.isDigit(stack.peek())) {
                    sb.insert(0, stack.pop());
                }
                int num = Integer.valueOf(sb.toString());
                for (int i = 0; i < num; i++) {
                    for (char ch : subString.toCharArray()) {
                        stack.push(ch);
                    }
                }
            }
        }
        StringBuffer sb = new StringBuffer();
        while (!stack.isEmpty()) {
            sb.insert(0, stack.pop());
        }
        return sb.toString();
    }

    int index = 0;//是全局的，因为递归回来之后是接着往下走，而非回来

    public String decodeString3(String s) {
        //每层递归的任务都一样，上一层读到左括号会开始下一层递归，新起一个sb，遇到字符直接拼接，遇到数字计算个数，然后下一个一定是左括号，然后开始下一层递归，返回本层括号内的字符内容
        StringBuffer sb = new StringBuffer();
        while (index < s.length()) {
            char c = s.charAt(index);
            //if(c=='[') 不可能以左括号开始，即使循环完成一次也是
            if (c == ']') {//遇到]说明是上一级的终止，返回
                index++;
                return sb.toString();
            } else if (Character.isDigit(c)) {
                int num = 0;
                while (Character.isDigit(s.charAt(index))) {
                    num = num * 10 + s.charAt(index) - '0';
                    index++;
                }//数字之后必定是[，此时index指向[
                index++;
                String nextS = decodeString3(s);
                for (int i = 0; i < num; i++) {
                    sb.append(nextS);
                }
            } else {//读到字符直接拼接
                index++;
                sb.append(c);
            }
        }
        return sb.toString();
    }

    public String decodeString4(String s) {
        StringBuilder decoded = new StringBuilder();
        int i = 0;
        while (i < s.length()) {
            if (Character.isDigit(s.charAt(i))) {
                int num = 0;
                while (Character.isDigit(s.charAt(i))) {
                    num = num * 10 + (s.charAt(i) - '0');
                    i++;
                }
                // start指向[后面的第一个字符
                int start = i++;
                int count = 1;
                while (count != 0) {
                    if (s.charAt(i) == '[') count++;
                    else if (s.charAt(i) == ']') count--;
                    i++;
                }//i指向本次递归中[对应的]的下一个字符
                String subStr = decodeString4(s.substring(start, i - 1));
                for (int j = 0; j < num; j++) {
                    decoded.append(subStr);
                }
            } else {
                decoded.append(s.charAt(i));
                i++;
            }
        }
        return decoded.toString();
    }

    public int[] dailyTemperatures1(int[] temperatures) {
        //两个栈，一个放温度一个放下标，比栈顶大就出栈，糊涂了，有下标不就有温度了吗
        int[] res = new int[temperatures.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < temperatures.length; i++) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int pre = stack.pop();
                res[pre] = i - pre;
            }
            stack.push(i);
        }
        return res;
    }

    public int[] dailyTemperatures(int[] temperatures) {
        //从右向左计算，利用后面已经算出的结果，快速算前面
        int n = temperatures.length;
        int[] res = new int[n];//最后一个结果一定为零
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j += res[j]) {//j为第一个可能比i大的
                if (temperatures[i] < temperatures[j]) {
                    res[i] = j - i;
                    break;
                } else if (res[j] == 0) { //t[j]<=t[i],但是已经没有比t[j]更大的了
                    res[i] = 0;
                    break;
                }
            }
        }
        return res;
    }

    public int largestRectangleArea(int[] heights) {
        //在一维数组中对每一个数找到第一个比自己小的元素。这类“在一维数组中找第一个满足某种条件的数”的场景就是典型的单调栈应用场景。
        //遍历所有的位置，以其为高度，找左右第一个比他小的，然后计算出一个可能为结果的面接(right-left-i)*height
        //是一个严格递增栈，为什么一次遍历就能找到结果？一次遍历是怎么快速找到左右最小的
        Stack<Integer> stack = new Stack<>();
        int n = heights.length + 2;
        int res = 0;
        //相当于加了哨兵，不用处理栈空和最后仍有元素的情况
        int[] newHeight = new int[n];
        newHeight[0] = 0;
        newHeight[n - 1] = 0;
        for (int i = 1; i < n - 1; i++) {
            newHeight[i] = heights[i - 1];
        }
        heights = newHeight;
        stack.push(0);//先放左哨兵
        for (int i = 0; i < n; i++) {
            while (heights[stack.peek()] >= heights[i]) {//等于的话也要出栈 1.只有严格小才是边界，所以必须保证栈里严格增，2.第二个相等的找到的面积一定更大
                int h = heights[stack.pop()];//需要处理边界，如果栈里只有一个元素，pop出来为高，那左边界应该为-1
                int width = (i - stack.peek() - 1);
                res = Math.max(res, width * h);
            }
            stack.push(i);
        }
        return res;
    }


}
