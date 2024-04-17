public class test {
    public static void main(String[] args) {
        ListNode head = new ListNode(2);
        ListNode node2 = new ListNode(3);
        ListNode node3 = new ListNode(4);
        ListNode node4 = new ListNode(1);
        // 构建链表          2  3 4     1
        head.next = node2;
        node2.next = node3;
        node3.next = node4;
//          int[] nums = new int[]{5,4,9,1000,52,37,12,0};
//        int[] nums = new int[]{86,39,77,23,32,45,58,63,3,4,37,22};
//        int n = nums.length;
//          heapSort(nums);
//        System.out.println(Arrays.toString(nums));

        sortList(head);
    }
    public static ListNode sortList(ListNode head) {
        return partation(head);
    }
    public static ListNode partation(ListNode head){
        if(head==null|| head.next==null) return head;
        ListNode slow = head,fast = head;
        while (fast!=null&&fast.next!=null){
            slow = slow.next;
            fast = fast.next.next;
        }//slow为分割后第二个链表的头节点
        ListNode mid = slow.next;
        slow.next = null;
        ListNode l1 = partation(head);
        ListNode l2 = partation(mid);
        ListNode sorted = merge(l1,l2);
        return sorted;
    }
    //合：合并两个有序链表：递归(没写出来)，迭代
    public static ListNode merge(ListNode head1,ListNode head2){
        ListNode dummy = new ListNode(0);
        ListNode pre = dummy;
        ListNode l1 = head1,l2 = head2;
        while (l1!=null&&l2!=null){
            if(l1.val<=l2.val){
                pre.next = l1;
                l1 = l1.next;
            }else {
                pre.next = l2;
                l2 = l2.next;
            }
            pre = pre.next;
        }
        if(l1==null) pre.next = l2;
        if(l2==null) pre.next = l1;
        return dummy.next;

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

    public static void heapSort(int[] arr){
        int n = arr.length;
        //从最后一个非叶子节点开始往前找
        //构造一个最大堆
        for(int i= (n/2)-1;i>=0;i--){
            heapify(arr,n,i);
        }
//        排序，每次取出当前最大值放到尾部
        for(int i=n-1;i>0;i--){
            //现在根一定是最大值，交换根和最后一个元素，然后把最后一个元素拿出二叉树
            //当前的最后一个元素的索引就是i
            swap(arr,0,i);
            //交换后重新调整，构造二叉堆
            heapify(arr,i,0);
        }

    }
    public boolean isMatch(String s, String p) {
        //1.为什么要动态规划
        int m = s.length(),n = p.length();
        char[] ss = s.toCharArray();
        char[] pp = p.toCharArray();
        boolean[][] dp = new boolean[m][n];
        //初始化较复杂，要注意
        //a可以和*a匹配
            //p的前j个(0,j-1)能不能构成s的前i个
            for(int i=0;i<m;i++){
                for (int j=0;j<n;j++){
                    if(pp[j]==ss[i] || pp[j]=='.') dp[i][j] = dp[i-1][j-1];
                    else if(pp[j]=='*'){
                        if(pp[j-1]=='.'||pp[j-1]==ss[i]) dp[i][j] = dp[i-1][j-1];
                        dp[i][j] =dp[i][j] || dp[i][j-1];//没代表任何字符
                    }else dp[i][j] = false;
                }
            }
        return dp[m][n];
    }

    public static void heapify(int[] arr,int n,int i){
        int largest = i;
        //找到i的左右子节点
        int l = 2*i+1;
        int r = 2*i+2;
        if(l<n && arr[largest] < arr[l]){
            largest = l;
        }
        if(r<n && arr[largest] < arr[r]){
            largest = r;
        }
        if(largest != i){
            swap(arr,largest,i);
            //交换后，被换下去那个节点处在一个新的位置，为了保证最大堆，还需要调整他的子树
            heapify(arr, n, largest);
        }
    }
    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
