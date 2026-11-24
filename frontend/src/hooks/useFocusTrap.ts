import { useEffect, useRef, useCallback } from 'react';

export interface UseFocusTrapOptions {
  /**
   * Whether the focus trap is currently active.
   */
  isActive: boolean;
  /**
   * Callback invoked when Escape key is pressed.
   */
  onClose?: () => void;
  /**
   * Optional ref to the element that triggered the dialog, to explicitly restore focus on close.
   * If omitted, falls back to document.activeElement captured immediately prior to trap activation.
   */
  triggerRef?: React.RefObject<HTMLElement | null>;
  /**
   * Optional ref to the element that should receive initial focus when the trap activates.
   * Defaults to the first focusable element within the container.
   */
  initialFocusRef?: React.RefObject<HTMLElement | null>;
  /**
   * Whether Escape key should trigger onClose. Defaults to true.
   */
  enableEscape?: boolean;
  /**
   * Whether to make application content outside the modal container inert while active.
   * Restores the exact previous inert state on close or unmount without blindly unsetting
   * elements that were already inert before opening. Defaults to false.
   */
  setInert?: boolean;
}

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

/**
 * Lightweight, zero-dependency focus management hook for accessible modal dialogs and navigation drawers.
 *
 * Implements WCAG 2.2 Level AA compliance:
 * - Moves initial focus into the dialog on activation.
 * - Traps Tab and Shift+Tab cycling strictly within the dialog boundaries.
 * - Supports Escape key dismissal.
 * - Automatically restores focus to the triggering element upon closure.
 */
export function useFocusTrap<T extends HTMLElement = HTMLDivElement>({
  isActive,
  onClose,
  triggerRef,
  initialFocusRef,
  enableEscape = true,
  setInert = false,
}: UseFocusTrapOptions) {
  const containerRef = useRef<T | null>(null);
  const previousActiveElementRef = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  const getFocusableElements = useCallback((): HTMLElement[] => {
    if (!containerRef.current) return [];

    const elements = Array.from(
      containerRef.current.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)
    );

    return elements.filter((el) => {
      // Filter out hidden, zero-size, or non-visible elements
      const isVisible = el.offsetWidth > 0 || el.offsetHeight > 0 || el.getClientRects().length > 0;
      const isNotHidden = el.getAttribute('aria-hidden') !== 'true';
      return isVisible && isNotHidden;
    });
  }, []);

  // Handle Tab and Escape keyboard navigation
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!isActive) return;

      // Handle Escape dismissal
      if (enableEscape && e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        onCloseRef.current?.();
        return;
      }

      // Handle Tab containment
      if (e.key === 'Tab') {
        const container = containerRef.current;
        if (!container) return;

        const focusables = getFocusableElements();

        if (focusables.length === 0) {
          e.preventDefault();
          container.focus();
          return;
        }

        const firstElement = focusables[0];
        const lastElement = focusables[focusables.length - 1];
        const activeEl = document.activeElement;

        if (e.shiftKey) {
          // Shift + Tab: backward navigation
          if (activeEl === firstElement || !container.contains(activeEl)) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          // Tab: forward navigation
          if (activeEl === lastElement || !container.contains(activeEl)) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      }
    },
    [isActive, enableEscape, getFocusableElements]
  );

  useEffect(() => {
    if (!isActive) return;

    // 1. Capture the element that had focus before opening
    if (typeof document !== 'undefined') {
      previousActiveElementRef.current = (triggerRef?.current || document.activeElement) as HTMLElement | null;
    }

    // 2. Manage background inert state if setInert is enabled
    // Store Map of elements and their previous inert state: Map<HTMLElement, boolean>
    const inertMap = new Map<HTMLElement, boolean>();

    const applyInertToBackground = (container: HTMLElement) => {
      if (typeof document === 'undefined') return;

      let current: HTMLElement | null = container;
      while (current && current !== document.body && current.parentElement) {
        const parent: HTMLElement = current.parentElement;
        for (let i = 0; i < parent.children.length; i++) {
          const sibling = parent.children[i] as HTMLElement;
          if (sibling !== current && sibling.nodeType === 1) { // Node.ELEMENT_NODE
            if (!inertMap.has(sibling)) {
              // Record exact previous state: check both attribute and property
              const wasInert = sibling.hasAttribute('inert') || (sibling as any).inert === true;
              inertMap.set(sibling, wasInert);

              sibling.setAttribute('inert', '');
              try {
                (sibling as any).inert = true;
              } catch {
                // Safe fallback for environments with partial inert support
              }
            }
          }
        }
        current = parent;
      }
    };

    if (setInert && containerRef.current) {
      applyInertToBackground(containerRef.current);
    }

    // 3. Move initial focus into the container
    const initialTimer = setTimeout(() => {
      if (!containerRef.current) return;

      if (setInert && inertMap.size === 0) {
        applyInertToBackground(containerRef.current);
      }

      if (initialFocusRef?.current && typeof initialFocusRef.current.focus === 'function') {
        initialFocusRef.current.focus();
        return;
      }

      const focusables = getFocusableElements();
      if (focusables.length > 0) {
        focusables[0].focus();
      } else {
        containerRef.current.focus();
      }
    }, 20);

    // 4. Attach keyboard listener
    window.addEventListener('keydown', handleKeyDown, true);

    return () => {
      clearTimeout(initialTimer);
      window.removeEventListener('keydown', handleKeyDown, true);

      // Restore exact previous inert state
      for (const [element, wasInert] of inertMap.entries()) {
        if (wasInert) {
          // Do not blindly set inert=false if the element was already inert before opening!
          element.setAttribute('inert', '');
          try {
            (element as any).inert = true;
          } catch {}
        } else {
          element.removeAttribute('inert');
          try {
            (element as any).inert = false;
          } catch {}
        }
      }
      inertMap.clear();

      // Restore focus on close / unmount
      try {
        const elementToRestore = triggerRef?.current || previousActiveElementRef.current;
        if (
          elementToRestore &&
          typeof elementToRestore.focus === 'function' &&
          typeof document !== 'undefined' &&
          document.body &&
          document.body.contains(elementToRestore)
        ) {
          setTimeout(() => {
            try {
              if (document.body && document.body.contains(elementToRestore)) {
                elementToRestore.focus();
              }
            } catch {
              // Gracefully handle focus failure if element detached during timeout
            }
          }, 10);
        }
      } catch {
        // Fail gracefully without crashing if trigger element is missing/invalid
      }
      previousActiveElementRef.current = null;
    };
  }, [isActive, setInert, handleKeyDown, initialFocusRef, triggerRef, getFocusableElements]);

  return containerRef;
}

export default useFocusTrap;
