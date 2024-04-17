package top;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

public class L140toL160 {
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
    /**
     * 141.判断链表是否有环:双指针，或者set判重
     * 142.是否有环2：找到环处的节点：原理没搞懂，也可用set
     * 143.重排链表:拆分，倒序，合并 或者 List<ListNode>链表转线性表
     * 144.遍历二叉树
     * 145.遍历二叉树
     * 146.LRU:自定义双向链表，注意规范化
     * 147.对链表插入排序：o(n^2)，细节还是没处理好，写错了
     * 148.链表排序：o(nlogn)应该是归并，但是想的是快排
     * 149.
     * 150. 逆波兰表达式求值：栈+字符串处理，判断是否为加减乘除，如何转数字,switch语法
     * 151.翻转字符串中的单词，注意split的用法，本题应该用双指针，还有trim
     * 152.乘积最大子数组:动态规划，注意细节
     * 153.寻找旋转排序数组中的最小值：想错了，
     * 154.
     * 155.最小栈：法1，搞两个栈 法2：一个栈，没想出来
     * 156.
     * 157.
     * 158.
     * 159.
     * 160.相交链表，没理透，没写出来
     * 162.寻找峰值，要求logn,二分，没做出来，核心就是往上升的方向二分
     * 164.最大间距，要求「线性时间」内运行并使用「线性额外空间」的算法：桶排序/基数排序
     * 162
     */
    public boolean hasCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast!=null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(slow==fast) return true;
        }
        return false;
    }
    public ListNode detectCycle(ListNode head) {
        //第一次相遇，快(k)比慢(m)多走n圈  k=2*m k=m+nb m=nb(即这时慢走了n圈)
        //初始节点的位置：a+nb,因此只需要再让慢走a步
        ListNode slow = head;
        ListNode fast = head;
        while (fast!=null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(slow==fast) {
                ListNode third = head;
                while (slow!=third){
                    slow = slow.next;//一定是slow
                    third = third.next;
                }
                return third;
            }
        }
        return null;
    }
    public void reorderList(ListNode head) {
        //拆成两个，把第二个反转
        ListNode dummy = new ListNode(0,head);
        ListNode fast = head,slow = head;
        while (fast.next!=null&&fast.next.next!=null){//3个节点要第二个，4个节点也要第二个
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode head2 = slow.next;
        slow.next = null;
        head2 = reveseList(head2);
        while (head2!=null){//怎么把两个链表连起来？三指针（每次分别从1和2拿一个节点放到cur后面），不好，不如直接改
            ListNode temp1 = head.next;
            ListNode temp2 = head2.next;
            head.next = head2;
            head2.next = temp1;
            head = temp1;
            head2 = temp2;
        }
    }
    public ListNode reveseList(ListNode head){
        if(head.next==null) return head;
        ListNode newHead = reveseList(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }
    static class LRUCache {
        //解决插入过程中，如果map满了，要移除哪一个个
        static class DLinkedNode{
            int key;//为什么一定要有key，因为超过容量的时候，从dlist中找到最后一个节点，然后需要根据这个节点反向去删除map，所以需要能够保证，从节点能找到map的key
            int val;
            DLinkedNode prev;
            DLinkedNode next;
            public DLinkedNode() {}
            public DLinkedNode(int key,int val) {
                this.key = key;
                this.val = val;
            }
        }
        private DLinkedNode head;//必须在外面声明，函数里创建的话外部用不了
        private DLinkedNode tail;//细节private
        int capacity;
        int count;
        Map<Integer,DLinkedNode> map = new HashMap<>();//map里为什么不直接存值？存值没法从双向链表中定位并移动
        public LRUCache(int capacity) {
            head = new DLinkedNode();//别忘了要new创建
            tail = new DLinkedNode();//别忘
            head.next = tail;
            tail.prev = head;
            this.count = 0;//最好初始化的时候赋值
            this.capacity = capacity;
        }
        public int get(int key) {
            DLinkedNode node = map.get(key);
            if(node==null) return -1;
            moveToHead(node);
            return node.val;
        }
        public void put(int key, int value) {
            //一定是会放进去的，超容量会移除一个旧的
            DLinkedNode node = map.get(key);
            if(node!=null) {
                node.val = value;
                moveToHead(node);
                return;
            }
            //新建并插入，还要看是否超容量
            node = new DLinkedNode(key,value);
            map.put(key,node);
            addToHead(node);
            count++;
            if(count>capacity){
                //删除最后一个
                DLinkedNode delete = delete();
                map.remove(delete.key);
                count--;
            }
        }
        //超容量时移除最后一个
        public DLinkedNode delete(){
            DLinkedNode node = tail.prev;
            node.prev.next = tail;
            tail.prev = node.prev;
            return node;//返回节点用于从map中移除
        }
        //把一个新的节点加到头
        public void addToHead(DLinkedNode node){
            node.next = head.next;
            node.next.prev = node;
            node.prev = head;
            head.next = node;
        }
        //把老的移到头
        public void moveToHead(DLinkedNode node){
            node.prev.next = node.next;
            node.next.prev = node.prev;
            addToHead(node);
        }

    }
    public ListNode insertionSortList(ListNode head) {
        if(head==null) return head;
        //比当前元素和前一个节点，大的话不动，小的话从头开始找，找到一个位置插入
        ListNode dummy = new ListNode(0,head);
        ListNode pre = head;
        ListNode cur = head.next;
        while (cur!=null){
            if(cur.val>=pre.val){
                pre = pre.next;
                //cur = pre.next;
            }
            else {//如果当前节点需要向前移动，pre其实是不需要动的
                ListNode temp = dummy;
                while (temp.next.val<cur.val){
                    temp = temp.next;
                }
                //cur插到temp后面
                pre.next = cur.next;//别忘
                cur.next = temp.next;
                temp.next = cur;
            }
            cur = pre.next;
        }
        return dummy.next;
    }
    public ListNode sortList(ListNode head) {
        //快排或归并，感觉本题应该用快排，因为归并没法记录子链表？
        //快排：两个函数，都需要left和right索引确定当前要排的是哪部分
        //归并：数组的话是数组拷贝，链表的话，只需要找到中间节点即可，所以本题更适合归并，注意要求o（1）空间，所以应该选则自底向上归并
        if(head==null||head.next==null) return head;
        ListNode slow =head,fast = head;
        while (fast.next!=null && fast.next.next!=null){
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode mid = slow.next;
        slow.next = null;
        head = sortList(head);
        mid = sortList(mid);
        return merge(head,mid);
    }
    //合并两个有序链表,可以递归可以迭代，但要求o(1)空间，所以只能迭代
    public ListNode merge(ListNode head1, ListNode head2){
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (head1!=null&&head2!=null){
            if(head1.val<head2.val){
                cur.next = head1;
                head1 = head1.next;
            }else{
                cur.next = head2;
                head2 = head2.next;
            }
            cur = cur.next;
        }
        cur.next = head1==null? head2:head1;
        return dummy.next;
    }
    //todo 149
    public int evalRPN(String[] tokens) {
        //用栈：遇到数字则入栈；遇到算符则取出栈顶两个数字进行计算，并将结果压入栈中
        Stack<Integer> stack = new Stack<>();
        for(String token: tokens){
            if(!("+".equals(token)||"-".equals(token)||"*".equals(token)||"/".equals(token))){
                stack.push(Integer.parseInt(token));
            }else{
                int num2 = stack.pop();//先出的是右值
                int num1 = stack.pop();
                switch (token){
                    case "+" :
                        stack.push(num1+num2);
                        break;//break用于防止case穿透
                    case "-":
                        stack.push(num1-num2);
                        break;
                    case "*":
                        stack.push(num1*num2);
                        break;
                    case  "/":
                        stack.push(num1/num2);
                        break;
                    default:
                }
            }
        }
        return stack.pop();
    }
    public String reverseWords(String s) {
        //为什么不能用split？比如a空空空b分割后会是a空b
        s = s.trim();
        int i=s.length()-1;
        StringBuilder sb = new StringBuilder();
        while (i>=0){
            while (i>=0 && s.charAt(i)==' ') i--;
            int j = i;
            while (i>=0 && s.charAt(i)!=' ') i--;
            sb.append(s.substring(i+1,j+1));
            sb.append(" ");
        }
        return sb.toString().trim();//因为最后一个sb也被拼接空格了
    }
    public int maxProduct(int[] nums) {
        int n = nums.length;
        int[] max = new int[n];
        int[] min = new int[n];
        max[0] = nums[0]; min[0] = nums[0]; int res = nums[0];
        for(int i = 1;i<n;i++){
            max[i] = Math.max(max[i-1]*nums[i],Math.max(min[i-1]*nums[i],nums[i]));//别忘了还有nums[i]本身也可能
            res = Math.max(res,max[i]);
            min[i] = Math.min(max[i-1]*nums[i],Math.min(min[i-1]*nums[i],nums[i]));
        }
        return res;//不是max[n-1];
    }
    public int findMin(int[] nums) {
        //想图像，想旋转二分查找的思路，怎么确定要找的值在哪一部分,区别在于不一定从有序那部分继续找而是要找最小的
        int left = 0,right = nums.length-1;
        while (left<right){//只剩一个元素没必要进入，一定是最小值
            int mid = (left+right)/2;
            //四种情况，mid应该和left比还是right，还是两者都可以
            //想图像：左值 < 中值, 中值 < 右值 2.左值 > 中值, 中值 < 右值 3.左值 < 中值, 中值 > 右值 4.左值 > 中值, 中值 > 右值（不可能）
            //一定是和heigh比，因为最后留下的可能会是一个升序段，z字形和升序处理left的流程是不一样的
            if(nums[mid]<nums[right]) right = mid;//把小的留下，很好理解
            else left = mid+1;//只有z字形可能出现mid大于right，mid等于right只有在left等于right的时候，但是这个没进循环
        }
        return nums[left];//right也行
    }
    class MinStack {
        //怎么记录每个元素对应的最小元素？1.再来一个栈
        //如果只用一个变量记录最小值，如果栈中这个元素出栈，前一个最小值怎么找？栈里记录当前元素减前一个元素，int变量记录当前元素
        // 设计三个元素，premin,preval,val=min,栈里存preval-premin,val-premin
        Stack<Long> stack;
        long min;
        public MinStack() {
            stack = new Stack<>();
        }
        public void push(int val) {
            if(stack.isEmpty()){
                stack.push(0L);
                min = val;
                return;
            }
            stack.push(val-min);
            if(val<min) min = val;
        }
        public void pop() {
            long pop = stack.pop();
            if(pop<0){
                min = min - pop;
            }
        }
        public int top() {
            if(stack.peek()<0) return (int)min;
            return (int)(stack.peek()+min);
        }
        public int getMin() {
            return (int)min;
        }
    }
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        //同一位置出发，相遇就是所要节点，通过双指针消除步行差，相当于在a链前补了一个b链，b链前补了一个a链，然后同时出发
        // 黄蓝+红蓝 红蓝+黄蓝 同时出发会同时到达第二个蓝
        ListNode pa = headA,pb = headB;
        while (pa!=pb){
            pa = pa==null? headB : pa.next;//null必须也得走，要不两链表不相交的话没法触发a=b
            pb = pb==null? headA : pb.next;
        }
        return pa;
    }
    //162 寻找峰值
    public int findPeakElement(int[] nums) {
        //题里给了左右两侧认为无穷小，所以问题转换为找最大值，或边界
        //往递增的方向上，二分  二分的重点在于二段性而非单调性
        int left = 0 ,right = nums.length-1;
        while (left<right){//一定有解，所以小于就行
            int mid = (left+right)/2;
            if(nums[mid]>nums[mid+1]) right = mid;//while处没取等，否则必须mid-1
            else left = mid+1;//nums[mid]<nums[mid+1]

        }
        return right;
    }


}
