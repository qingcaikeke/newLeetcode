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
    /**
     * 哈希
     * 1.两数之和：秒
     * 2.字母异位词分组：没给哈希提示肯定想不到，还得练
     * 3.最长连续序列：set，没写明白
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
     * 链表
     * 22.相交链表：双指针
     * 23.反转链表：递归迭代，递归反转链表会消耗额外空间
     * 24.回文链表：后半反转
     * 25.环形链表：快慢指针，扣圈
     * 26.环形链表2：第一次相遇后，起第三个指针和慢一起走，再次相遇为入环节点，关键在于证明第一次相遇慢走了n圈
     * 27.合并两个有序链表:递归迭代
     * 28.两数相加：挺熟了
     * 29.删除倒数第n个节点：双指针，注意dummy防止删除的是第一个
     * 30.两两交换节点：递归迭代，略生疏,
     * 31.k个一组翻转链表：模拟，查k个翻转就行了，注意细节
     * 32.随机链表的复制：顺序拷，dfs，bfs，省略map空间
     * 33.排序链表：插入或归并，还有个自底向上的思路，了解一下（先把12合并，再把34合并...再上一层，把1234合并）
     * 34.合并k个升序链表：归并，或者用一个优先级队列（底层是最小堆，但是堆不用自己实现）
     * 35.lru缓存：写了很多次了
     * 二叉树
     * 36.中序遍历：递归迭代morris,基本功，但还是出错了
     * 37.二叉树深度:递归秒，可广度
     * 38.翻转二叉树:递归秒，可广度
     * 39.对称二叉树：注意判定方法，不是左子val=右子val就是对称的。注意迭代解法要会（广度）
     * 40.二叉树直径:dfs
     * 41.层序遍历:广度秒
     * 42.有序数组转二叉搜索树：递归，有点分治的感觉
     * 43.验证二叉搜索树:中序遍历（递归、迭代）或前序遍历（递归），递归居然也能做，没想到
     * 44.二叉搜素树中第k小的元素:基本中序遍历，知道进阶思路：频繁插入删除，频繁查第k小，如何优化
     * 45.二叉树的右视图:bfs加每层最后一个节点，dfs注意不能只看右子树，有可能左比右深，所以需要维护一个深度
     * 46.二叉树展开为链表：morris最简单，但也没写好；借助栈，没想到；逆前序遍历，没想明白
     * 47.从前序与中序遍历序列构造二叉树:递归
     * 48.路径总和3：大概想出了一个前缀和思想，之后就不会了,二叉树遍历，结合前缀和结合map
     * 49.二叉树的最近公共祖先，还是没做出来，法2：储存根到p的路径和根到q的路径，然后比较两条路径，可以用map记录每个节点的根节点
     * 50.二叉树最大路径和，还行
     */
    public void moveZeroes(int[] nums) {
        int index = 0;
        for(int i=0;i<nums.length;i++){
            if(nums[i]!=0){
                swap(nums,i,index);
                index++;
            }
        }
    }
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        int pLen = p.length();
        if(pLen>s.length()) return res;
        int[] need = new int[26];
        int[] window = new int[26];
        for(int i=0;i<pLen;i++){
            need[p.charAt(i)-'a']++;
            window[s.charAt(i)-'a']++;
        }
        if(Arrays.equals(need,window)) res.add(0);
        for(int i=0;i<s.length()-pLen;i++){
            window[s.charAt(i)-'a']--;
            window[s.charAt(i+pLen)-'a']++;
            if(Arrays.equals(window,need)) res.add(i+1);
        }
        return res;
    }
    public int subarraySum(int[] nums, int k) {
        //1.暴力，找所有子数组o(n^3)(子数组n^2,遍历子数组求和o(n)) 所以想到
        // 遍历右边界的同时求和，这样就降了一维，想到该思路类似前缀和，前缀和也可以降一维
        // 前缀和加哈希表o(n)，哈希表记录所需前缀和是否出现过及出现次数
        //感觉双指针滑动窗口可以，试一下，可能有负数，没法滑动窗口，同时滑动窗口会漏0
        int preSum=0;int res=0;
        Map<Integer,Integer> map = new HashMap<>();
        map.put(0,1);//别忘，前缀和为0的出现次数为1
        for(int i=0;i<nums.length;i++) {
            preSum+=nums[i];
            if(map.containsKey(preSum-k)) res+=map.get(preSum-k);//先if，因为k可能得0
            map.put(preSum,map.getOrDefault(preSum,0)+1);

        }
        return res;
    }
    public int[] maxSlidingWindow(int[] nums, int k) {
        int[] res = new int[nums.length-k+1];//起始：0到等于n-k
        Deque<Integer> deque = new LinkedList<>();//维护的是一个单调非增队列
        //后入,如果前面的元素比入的还小，那他就不可能是最大值，因为窗口中一定包含最后入的这个元素
        //怎么判断队头已经到时间了，需要出队了,根据deque中存的下标
        for(int i=0;i<nums.length;i++){
            while (!deque.isEmpty() && nums[deque.peekLast()]<nums[i]){//如果相等也会入队
                deque.pollLast();
            }
            deque.addLast(i);
            //计算以i结尾的窗口的最大值
            if(deque.peekFirst()==i-k) deque.pollFirst();
            if(i>=k-1) res[i-k+1] = nums[deque.peekFirst()];
        }
        return res;
    }
    public String minWindow1(String s, String t) {
        //要求o(m+n) 时间
        Map<Character,Integer> need = new HashMap<>();
        Map<Character,Integer> window = new HashMap<>();
        for(char c: t.toCharArray()){
            need.put(c, need.getOrDefault(c,0)+1);
        }
        int resLeft=-1,resLen= Integer.MAX_VALUE;
        int l=0,r=0;
        while (r<s.length()){
            window.put(s.charAt(r),window.getOrDefault(s.charAt(r),0)+1);
            //比较两个map，检查是否包含全部字串了
            while (l<=r && check(need,window)){
                int curLen = r-l+1;
                if(curLen<resLen){
                    resLeft = l;
                    resLen = curLen;
                }
                window.put(s.charAt(l),window.get(s.charAt(l))-1);
                l++;
            }
            r++;
        }
        return resLeft==-1? "" : s.substring(resLeft,resLeft+resLen);
    }
    public boolean check(Map<Character,Integer> need,Map<Character,Integer> window){
        Set<Map.Entry<Character, Integer>> entries = need.entrySet();
        for(Map.Entry<Character, Integer> entry : entries){
            char key = entry.getKey();
            if(window.getOrDefault(key,0)<entry.getValue()){
                return false;
            }
        }
        return true;
    }
    public String minWindow2(String s, String t){
        //优化为一个map，优化为根据valid判断窗口是否满足要求
        Map<Character ,Integer> map = new HashMap<>();
        //也可以用int[] need = new int[128]; need[t.charAt(i)]++;这个时候char代表的是ascll码
        for(char c:t.toCharArray()){
            map.put(c,map.getOrDefault(c,0)+1);
        }
        int l=0,r=0; int resLeft=-1,resLen=Integer.MAX_VALUE;
        int valid=0;//用于判断是否包含了t中所有字符
        while (r<s.length()){
            char c = s.charAt(r);
            if(map.containsKey(c)){
                map.put(c,map.get(c)-1);
                if(map.get(c)>=0) valid++;//如果当前这个元素还需要，就给valid++
            }
            while (l<=r && valid==t.length()){ //不用valid可以像上面一样，遍历map，看每个val是否都小于等于0
                int curLen = r-l+1;
                if(curLen<resLen){
                    resLeft = l;
                    resLen = curLen;
                }
                //移除一个left
                char cLeft = s.charAt(l);
                if(map.containsKey(cLeft)){
                    map.put(cLeft,map.get(cLeft)+1);
                    if(map.get(cLeft)>0) valid--;
                }
                l++;
            }
            r++;
        }
        return resLeft==-1? "" : s.substring(resLeft,resLeft+resLen);
    }
    public void rotate(int[] nums, int k) {
        //怎么达到o（1）空间？三次翻转，没想到，一个整体的思想
        //nums[i] -> nums[i+k%n]
        int n = nums.length;
        k = k%n;
        int count=0;
        int start=0;
        //跳着改变元素，count记录已经改变的个数
        while (count<n){
            int curIndex = start;
            int prev=nums[start];
            do{
                int nextIndex = (curIndex + k) % n;
                int temp = nums[nextIndex];
                nums[nextIndex] = prev;
                count++;
                prev = temp;
                curIndex = nextIndex;
            }while (curIndex != start);//必须do while 否则第一次进不了循环
            start++;
        }
    }
    public int[] productExceptSelf(int[] nums) {
        //不让用除法，而且除法要注意可能有0
        //法1：开两个数组分别储存左积和右积---法2：直接使用res，先从左到右记录左积，再从右到左把右积乘进去----法3双指针，同时进行左到右和右到左
        int n = nums.length;
        int[] res = new int[n];
        res[0] =1;
        for(int i =1;i<n;i++){
            res[i] = res[i-1] * nums[i-1];
        }
        int r=1;
        for(int j=n-1;j>=0;j--){
            res[j] = res[j+1] * r;
            r = r*nums[j];
        }
        return res;
    }
    public int firstMissingPositive(int[] nums) {
        //如何o(n)+o(1)
        //可能是缺个大的 1235，也可能是缺个小的234
        //1.存哈希表，然后从1开始枚举，看表里有没有 o(n)+o(n)
        //2.不用哈希表，从1枚举，看数组有没有 o(n^2)+o(1)
        //重点：res已经是[1,N+1]中的数
        for(int i=0;i<nums.length;i++){
            //把nums[i]换到下标num[i-1]的位置
            //换过来的元素也需要处理
            while (nums[i]>0 && nums[i]<=nums.length && nums[i]!=nums[nums[i]-1]){
                swap(nums,i,nums[i]-1);
            }
        }
        //1出现了第一个位置一定被换成1，找到第一个不符合规则的就是缺失的最小正数
        for(int i=0;i<nums.length;i++){
            if(nums[i]!=i+1) return i+1;
        }
        return nums.length+1;
    }
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length,n = matrix[0].length;
        int i=0,j = n-1;//从右上或左下开始搜索都行，重点在于两个方向一个变小一个变大，从而一点点减小范围
        while (i<n && j>=0){//注意一定不能是三个if，第二个if完成后会进到第三个
            if(matrix[i][j]==target) return true;
            else if(matrix[i][j]>target) j--;
            else i++;
        }
        return false;
    }
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr!=null){
            ListNode temp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = temp;
        }
        return prev;
    }
    public boolean isPalindrome(ListNode head) {
        //法1：额外空间，存到数组里 法2：后半反转
        ListNode slow = head,fast =head;
        while (fast.next!=null && fast.next.next!=null){
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode head2 = slow.next;
        slow.next = null;
        head2 = reverseList(head2);
        while (head2!=null){
            if(head.val!= head2.val) return false;
            head =  head.next;
            head2 = head2.next;
        }
        return true;
    }
    public ListNode swapPairs(ListNode head){
        ListNode dummy = new ListNode(0,head);
        ListNode pre = dummy;
        ListNode cur = head;
        while (cur.next!=null){
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
    //模版写法：栈里放的是根
    //还有其他写法，比如前序遍历，可以把右节点入栈，也可以右左入栈；比如中序遍历，可以右根入栈
    //后续怎么遍历，什么时候打印值？法1：逆后续，根右左的顺序遍历，然后reverse
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();//栈里放的是根，用根去找右
        while (root!=null || !stack.isEmpty()){//注意不是先放一个root，因为要保证栈里没有null
            while (root!=null){
                stack.push(root);//先把检验过的不是null的放进去
                root = root.left;//再往左移
            }
            root = stack.pop();
            res.add(root.val);
            root = root.right;
        }
        return res;
    }
    //补一个后续遍历，较难
    public List<Integer> postorderTraversal(TreeNode root){
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> res = new ArrayList<>();
        TreeNode prev = null;//核心，因为有两次到达节点的过程，判断该节点的右节点是否访问过，才能确定是打值还是向右走
        while (!stack.isEmpty() || root!=null){
            while (root!=null){
                stack.push(root);
                root = root.left;
            }
            //栈中弹出的元素，左子节点一定是访问完了的，再看完右节点就可以打印根值
            root = stack.pop();
            //右节点已经访问过或没有右节点
            if(root.right==null || root.right == prev){
                res.add(root.val);
                prev = root;
                root = null;
            }//右节点没访问过
            else{
                stack.push(root);
                root = root.right;
            }
        }
        return res;
    }
    //38
    public TreeNode invertTree(TreeNode root) {
        if(root==null )return root;
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }
    public int kthSmallest(TreeNode root, int k) {
        //进阶：如果二叉搜索树经常被修改（插入/删除操作）并且你需要频繁地查找第 k 小的值，你将如何优化算法
        //创建一个数据结构，储存节点的同时储存当前节点的子树的节点数，可用一个map，然后二分的去判断往左走还是往右走
        //再进一步还可以把树记录节点数的同时变为avl树，防止搜索树退化为链表
        Stack<TreeNode> stack = new Stack<>();
        while (!stack.isEmpty() || root!=null){
            while (root!=null){
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            k--;
            if(k==0) return root.val;
            root = root.right;
        }
        return -1;
    }
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        dfsRightSide(res,root,0);
        return res;
    }
    public void dfsRightSide(List<Integer> res,TreeNode root,int depth){
        //根右左遍历，只有第一次遍历某层的节点才加入
        if(root==null){
            return;
        }
        if(res.size()<=depth) res.add(root.val);
        dfsRightSide(res,root.right,depth+1);
        dfsRightSide(res,root.left,depth+1);
    }
    public void flatten1(TreeNode root) {
        TreeNode cur = root;
        while (cur!=null){
            if (cur.left!=null){
                TreeNode node = cur.left;
                while (node.right!=null&&node.right!=cur){
                    node = node.right;
                }//node为左子树的最右节点
                //为什么比morris少了一层if，因为是先序遍历，node指向的是cur的right，这样就不会重复遍历
                node.right = cur.right;
                cur.right = cur.left;
                cur.left = null;
                cur = cur.right;
            }
            else {
               cur = cur.right;
            }
        }
    }
    TreeNode pre = null;
    public void flatten2(TreeNode root) {
        //维护一个全局变量pre，右左根遍历（反前序遍历），把当前节点的右节点连为pre，左节点置为null
        //要想明白他为什么不会丢节点，丢信息（因为根一直没动，是最后看到，所以左右一定都能找到）
        if(root==null){
            return;
        }
        flatten2(root.right);
        flatten2(root.left);
        root.right = pre;
        root.left = null;
        pre = root;
    }
    public void flatten3(TreeNode root){
        if(root==null) return;
        //为什么一定要有prev？
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            if(node.right!=null) stack.push(node.right);
            if(node.left!=null) stack.push(node.left);
            if(!stack.isEmpty())node.right = stack.peek();//一定不是node.right=node.left，因为node可能是叶节点
            node.left = null;
        }
    }
    public void flatten4(TreeNode root){
        if(root==null) return;
        TreeNode right = root.right;
        flatten4(root.left);
        root.right = root.left;
        root.left = null;
        while (root.right!=null){
            root = root.right;//左子树展平后的最后一个节点
        }
        flatten4(right);
        root.right = right;
    }
    Map<Long,Integer> map = new HashMap<>();//记录前缀和等于num的有多少个
    public int pathSum(TreeNode root, int targetSum) {
        //前缀和思想，存根到叶的和，为什么想到前缀和，
        // 因为最开始的想法是，遍历所有点起始，然后分别从起始点找所有路径
        //dfs1搜索所有节点，dfs2搜索节点往下的所有路径
        map.put(0L,1);
        return dfsPathSum(root,targetSum,0);
    }
    public int dfsPathSum(TreeNode root,int targetSum,long curSum){
        //本质上是从上往下，统计以每个节点为「路径结尾」的合法数量
        if(root==null){
            return 0;
        }
        //dfs，根左右，遍历顺序决定写法
        curSum+=root.val;//当前节点的前缀和
        int res = map.getOrDefault(curSum-targetSum,0);
        map.put(curSum,map.getOrDefault(curSum,0)+1);
        res+=dfsPathSum(root.left,targetSum,curSum);
        res+=dfsPathSum(root.right,targetSum,curSum);
        map.put(curSum,map.getOrDefault(curSum,0)-1);//与上面那个对应，左枝移到右枝，左对应那个前缀和应该剪掉，以保证在一条路径上
        return res;
    }
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        //想到，后续遍历？怎么确定要返回的根节点
        //题解先序遍历，root一定是一个根，但需要确定有没有更好的，
        // 分别去看左右子树有没有p，q，只要左子树出现，说明左子节点是一个解，直接返回，全看右子树还能不能再出现一个
        //左右都出现，说明只能是根节点
        if(root==null || root==p || root==q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if(left==null) return right;
        if(right==null) return left;
        return root;//返回值就是所找的最近父节点
    }

    
}
