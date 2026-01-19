# linked_list_operations.py
# Common LeetCode Linked List Operations in one place

class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return str(self.val)


# ---------- Helper Utilities ----------

def build_linked_list(values):
    """Convert Python list -> Linked List"""
    dummy = ListNode()
    curr = dummy
    for v in values:
        curr.next = ListNode(v)
        curr = curr.next
    return dummy.next


def linked_list_to_list(head):
    """Convert Linked List -> Python list"""
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


# ---------- Core LeetCode Operations ----------

def reverse_list(head):
    """Reverse a linked list (Iterative)"""
    prev = None
    curr = head

    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt

    return prev


def merge_two_sorted_lists(l1, l2):
    """Merge two sorted linked lists"""
    dummy = ListNode()
    curr = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next

    curr.next = l1 or l2
    return dummy.next


def add_two_numbers(l1, l2):
    """
    Add two numbers represented as reversed linked lists
    LeetCode #2
    """
    dummy = ListNode()
    curr = dummy
    carry = 0

    while l1 or l2 or carry:
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0

        total = v1 + v2 + carry
        carry = total // 10
        curr.next = ListNode(total % 10)

        curr = curr.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None

    return dummy.next


def remove_nth_from_end(head, n):
    """Remove Nth node from end"""
    dummy = ListNode(0, head)
    slow = fast = dummy

    for _ in range(n):
        fast = fast.next

    while fast.next:
        slow = slow.next
        fast = fast.next

    slow.next = slow.next.next
    return dummy.next


def detect_cycle(head):
    """Detect cycle using Floyd's algorithm"""
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False


def middle_node(head):
    """Return middle node of linked list"""
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow


def sort_list(head):
    """
    Sort linked list using merge sort
    LeetCode #148
    """

    if not head or not head.next:
        return head

    # Find middle
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None

    left = sort_list(head)
    right = sort_list(mid)

    return merge_two_sorted_lists(left, right)


# ---------- Example Usage ----------

if __name__ == "__main__":
    # Build lists
    l1 = build_linked_list([2, 4, 3])
    l2 = build_linked_list([5, 6, 4])

    print("Add Two Numbers:")
    print(linked_list_to_list(add_two_numbers(l1, l2)))  # [7,0,8]

    print("\nReverse List:")
    print(linked_list_to_list(reverse_list(build_linked_list([1,2,3,4]))))

    print("\nMerge Two Sorted Lists:")
    merged = merge_two_sorted_lists(
        build_linked_list([1,3,5]),
        build_linked_list([2,4,6])
    )
    print(linked_list_to_list(merged))

    print("\nRemove 2nd From End:")
    print(linked_list_to_list(remove_nth_from_end(
        build_linked_list([1,2,3,4,5]), 2
    )))

    print("\nMiddle Node:")
    print(middle_node(build_linked_list([1,2,3,4,5])).val)

    print("\nSort List:")
    print(linked_list_to_list(sort_list(
        build_linked_list([4,2,1,3])
    )))
