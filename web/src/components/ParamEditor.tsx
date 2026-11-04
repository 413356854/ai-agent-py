import { Form, Input, InputNumber, Select, Space, Typography } from 'antd'

type Params = Record<string, unknown>

interface FieldProps {
  value: unknown
  onChange: (next: unknown) => void
}

/** 递归渲染：数字 → 数字框；字符串数组 → 标签选择；对象 → 子字段。 */
function ScalarField({ value, onChange }: FieldProps) {
  if (typeof value === 'number') {
    return (
      <InputNumber
        value={value}
        min={0}
        style={{ width: 180 }}
        onChange={(next) => onChange(typeof next === 'number' ? next : 0)}
      />
    )
  }

  if (Array.isArray(value)) {
    return (
      <Select
        mode="tags"
        value={value as string[]}
        onChange={(next) => onChange(next)}
        style={{ width: '100%' }}
        tokenSeparators={[',', '，']}
        placeholder="输入后回车添加"
      />
    )
  }

  if (value !== null && typeof value === 'object') {
    return (
      <ObjectFields
        value={value as Record<string, unknown>}
        onChange={(next) => onChange(next)}
      />
    )
  }

  return <Input value={String(value ?? '')} onChange={(event) => onChange(event.target.value)} />
}

function ObjectFields({
  value,
  onChange,
}: {
  value: Record<string, unknown>
  onChange: (next: Record<string, unknown>) => void
}) {
  return (
    <Space direction="vertical" size={8} style={{ width: '100%' }}>
      {Object.entries(value).map(([key, item]) => (
        <div key={key} style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <Typography.Text style={{ width: 88, flex: '0 0 88px' }}>{key}</Typography.Text>
          <div style={{ flex: 1, minWidth: 0 }}>
            <ScalarField value={item} onChange={(next) => onChange({ ...value, [key]: next })} />
          </div>
        </div>
      ))}
    </Space>
  )
}

interface ParamEditorProps {
  value: Params
  onChange: (next: Params) => void
}

/** 参数编辑表单：只负责结构与取值，校验交给后端。 */
export function ParamEditor({ value, onChange }: ParamEditorProps) {
  return (
    <Form layout="vertical" size="small">
      {Object.entries(value).map(([key, item]) => (
        <Form.Item
          key={key}
          label={<Typography.Text code>{key}</Typography.Text>}
          style={{ marginBottom: 16 }}
        >
          <ScalarField value={item} onChange={(next) => onChange({ ...value, [key]: next })} />
        </Form.Item>
      ))}
    </Form>
  )
}
